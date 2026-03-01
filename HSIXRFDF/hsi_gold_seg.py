import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from spectral.io import envi

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def _extract_id(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    if not m:
        raise ValueError(f"Could not extract numeric id from {path}")
    return int(m.group(1))


def _read_envi(hdr_path: Path) -> np.ndarray:
    raw_path = hdr_path.with_suffix("")
    arr = np.asarray(envi.open(str(hdr_path), str(raw_path)).open_memmap())
    return arr


def _read_cube_npy(path: Path) -> np.ndarray:
    cube = np.load(path)
    if cube.ndim == 4 and cube.shape[0] == 1:
        cube = np.squeeze(cube, axis=0)
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D HSI cube, got {cube.shape} in {path}")
    return cube


def _read_cube(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return _read_cube_npy(path)
    if path.suffix.lower() in [".hdr", ".hsi", ".dat", ""]:
        arr = _read_envi(path if path.suffix else path.with_suffix(".hdr"))
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D ENVI cube, got {arr.shape} in {path}")
        return arr
    raise ValueError(f"Unsupported HSI file format: {path}")


def _normalize_hsi(cube: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    cube = cube.astype(np.float32)
    bmin = cube.min(axis=(0, 1), keepdims=True)
    bmax = cube.max(axis=(0, 1), keepdims=True)
    return (cube - bmin) / (bmax - bmin + eps)


def _select_even_bands(cube: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= cube.shape[2]:
        return cube
    idx = np.linspace(0, cube.shape[2] - 1, k).round().astype(int)
    return cube[:, :, idx]


def _preprocess_cube(cube: np.ndarray, drop_bands: int, num_bands: int) -> np.ndarray:
    if drop_bands > 0:
        cube = cube[:, :, drop_bands:]
    cube = _normalize_hsi(cube)
    cube = _select_even_bands(cube, num_bands)
    return cube


def _pad_if_needed(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr.shape[:2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    if arr.ndim == 3:
        return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant")


def discover_processed_pairs(processed_dir: Path) -> Dict[int, Tuple[Path, Path]]:
    hsi_files = {_extract_id(p): p for p in processed_dir.glob("hsi_bands10_pcb*.npy")}
    mask_files = {_extract_id(p): p for p in processed_dir.glob("gold_candidate_hsi_pcb*.png")}
    ids = sorted(set(hsi_files) & set(mask_files))
    return {i: (hsi_files[i], mask_files[i]) for i in ids}


def discover_general_pairs(dataset_root: Path) -> Dict[int, Tuple[Path, Path]]:
    hsi_root = dataset_root / "HSI"
    gm_root = hsi_root / "General_masks"

    pairs: Dict[int, Tuple[Path, Path]] = {}
    for pcb_dir in sorted(hsi_root.glob("pcb*")):
        if not pcb_dir.is_dir():
            continue
        if not pcb_dir.name[3:].isdigit():
            continue
        pid = int(pcb_dir.name[3:])
        cube_hdr = pcb_dir / f"pcb{pid}.hdr"
        mask_hdr = gm_root / f"{pid}.HDR"
        if cube_hdr.exists() and mask_hdr.exists():
            pairs[pid] = (cube_hdr, mask_hdr)
    return pairs


def train_val_split(ids: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    ids = list(ids)
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    val_count = max(1, int(round(len(ids) * val_ratio)))
    val_ids = sorted(ids[:val_count])
    train_ids = sorted(ids[val_count:])
    if not train_ids:
        train_ids = val_ids[:-1]
        val_ids = val_ids[-1:]
    return train_ids, val_ids


class GoldPatchDataset(Dataset):
    def __init__(
        self,
        pairs: Dict[int, Tuple[Path, Path]],
        ids: List[int],
        patch_size: int,
        samples_per_image: int,
        augment: bool,
        source: str,
        drop_bands: int,
        num_bands: int,
        positive_class: int,
    ) -> None:
        self.pairs = pairs
        self.ids = list(ids)
        self.patch_size = patch_size
        self.samples_per_image = samples_per_image
        self.augment = augment
        self.source = source
        self.drop_bands = drop_bands
        self.num_bands = num_bands
        self.positive_class = positive_class

    def __len__(self) -> int:
        return len(self.ids) * self.samples_per_image

    def __getitem__(self, idx: int):
        pcb_id = self.ids[idx % len(self.ids)]
        hsi_path, mask_path = self.pairs[pcb_id]

        cube = _read_cube(hsi_path).astype(np.float32)

        if self.source == "processed":
            mask = np.array(Image.open(mask_path), dtype=np.uint8)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            cube = np.clip(cube, 0.0, 1.0)
            mask = (mask > 0).astype(np.float32)
        else:
            mask_raw = _read_envi(mask_path)
            if mask_raw.ndim == 3:
                mask_raw = mask_raw[:, :, 0]
            mask = (mask_raw == self.positive_class).astype(np.float32)
            cube = _preprocess_cube(cube, drop_bands=self.drop_bands, num_bands=self.num_bands)

        cube = _pad_if_needed(cube, self.patch_size, self.patch_size)
        mask = _pad_if_needed(mask, self.patch_size, self.patch_size)

        h, w = cube.shape[:2]
        y0 = random.randint(0, h - self.patch_size)
        x0 = random.randint(0, w - self.patch_size)

        cube = cube[y0:y0 + self.patch_size, x0:x0 + self.patch_size, :]
        mask = mask[y0:y0 + self.patch_size, x0:x0 + self.patch_size]

        if self.augment:
            if random.random() < 0.5:
                cube = np.flip(cube, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
            if random.random() < 0.5:
                cube = np.flip(cube, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

        x = torch.from_numpy(np.transpose(cube, (2, 0, 1))).float()
        y = torch.from_numpy(mask[None, ...]).float()
        return x, y


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSmall(nn.Module):
    def __init__(self, in_channels: int, base: int = 32) -> None:
        super().__init__()
        self.down1 = ConvBlock(in_channels, base)
        self.down2 = ConvBlock(base, base * 2)
        self.down3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = ConvBlock(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))

        b = self.bottleneck(self.pool(d3))

        u3 = self.dec3(torch.cat([self.up3(b), d3], dim=1))
        u2 = self.dec2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.dec1(torch.cat([self.up1(u2), d1], dim=1))
        return self.out(u1)


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1, 2, 3))
    den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    return 1.0 - (num / den).mean()


def iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    inter = (pred * targets).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
    return float(((inter + eps) / (union + eps)).mean().item())


@dataclass
class TrainConfig:
    out_dir: Path
    epochs: int
    batch_size: int
    lr: float
    patch_size: int
    train_samples_per_image: int
    val_samples_per_image: int
    val_ratio: float
    seed: int
    num_workers: int
    source: str
    processed_dir: Path
    dataset_root: Path
    positive_class: int
    drop_bands: int
    num_bands: int


def train(cfg: TrainConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.source == "processed":
        pairs = discover_processed_pairs(cfg.processed_dir)
    else:
        pairs = discover_general_pairs(cfg.dataset_root)

    if len(pairs) < 2:
        raise RuntimeError("Need at least 2 paired samples to train/validate.")

    ids = sorted(pairs.keys())
    channel_counts: Dict[int, int] = {}
    skipped: List[Tuple[int, str]] = []
    valid_ids: List[int] = []
    pos_ratios: List[float] = []

    for pid in ids:
        hsi_path, mask_path = pairs[pid]
        try:
            cube = _read_cube(hsi_path)
            if cfg.source == "general":
                cube = _preprocess_cube(cube, cfg.drop_bands, cfg.num_bands)
                mask = _read_envi(mask_path)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                pos_ratios.append(float((mask == cfg.positive_class).mean()))
            ch = int(cube.shape[2])
            channel_counts[ch] = channel_counts.get(ch, 0) + 1
            valid_ids.append(pid)
        except Exception as exc:
            skipped.append((pid, str(exc)))

    if len(valid_ids) < 2:
        raise RuntimeError("Not enough valid samples after loading and preprocessing.")

    in_channels = max(channel_counts.items(), key=lambda kv: kv[1])[0]
    filtered_ids: List[int] = []
    for pid in valid_ids:
        cube = _read_cube(pairs[pid][0])
        if cfg.source == "general":
            cube = _preprocess_cube(cube, cfg.drop_bands, cfg.num_bands)
        if cube.shape[2] == in_channels:
            filtered_ids.append(pid)
        else:
            skipped.append((pid, f"channel mismatch: expected {in_channels}, got {cube.shape[2]}"))

    if len(filtered_ids) < 2:
        raise RuntimeError("Not enough compatible samples after channel filtering.")

    if skipped:
        print("Skipped samples:")
        for pid, reason in skipped:
            print(f"  pcb{pid}: {reason}")

    if cfg.source == "general" and pos_ratios:
        print(f"Positive-class ({cfg.positive_class}) pixel ratio avg={np.mean(pos_ratios):.5f}, min={np.min(pos_ratios):.5f}, max={np.max(pos_ratios):.5f}")

    train_ids, val_ids = train_val_split(filtered_ids, cfg.val_ratio, cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = GoldPatchDataset(
        pairs=pairs,
        ids=train_ids,
        patch_size=cfg.patch_size,
        samples_per_image=cfg.train_samples_per_image,
        augment=True,
        source=cfg.source,
        drop_bands=cfg.drop_bands,
        num_bands=cfg.num_bands,
        positive_class=cfg.positive_class,
    )
    val_ds = GoldPatchDataset(
        pairs=pairs,
        ids=val_ids,
        patch_size=cfg.patch_size,
        samples_per_image=cfg.val_samples_per_image,
        augment=False,
        source=cfg.source,
        drop_bands=cfg.drop_bands,
        num_bands=cfg.num_bands,
        positive_class=cfg.positive_class,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))

    model = UNetSmall(in_channels=in_channels).to(device)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.out_dir / "best_gold_hsi_unet.pt"
    best_val_iou = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = 0.5 * bce(logits, y) + 0.5 * dice_loss_from_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = 0.5 * bce(logits, y) + 0.5 * dice_loss_from_logits(logits, y)
                val_loss += loss.item()
                val_iou += iou_from_logits(logits, y)
        val_loss /= max(1, len(val_loader))
        val_iou /= max(1, len(val_loader))

        print(f"Epoch {epoch:03d}/{cfg.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_iou={val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "in_channels": in_channels,
                    "patch_size": cfg.patch_size,
                    "source": cfg.source,
                    "positive_class": cfg.positive_class,
                    "drop_bands": cfg.drop_bands,
                    "num_bands": cfg.num_bands,
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "best_val_iou": best_val_iou,
                },
                ckpt_path,
            )
            print(f"Saved new best checkpoint to: {ckpt_path}")

    print("Training complete.")
    print(f"Best val IoU: {best_val_iou:.4f}")


def _infer_tiled(model: nn.Module, cube: np.ndarray, device: torch.device, tile: int, overlap: int) -> np.ndarray:
    cube = np.clip(cube.astype(np.float32), 0.0, 1.0)
    h, w, _ = cube.shape
    stride = max(1, tile - overlap)

    prob_acc = np.zeros((h, w), dtype=np.float32)
    cnt_acc = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if h > tile and ys[-1] != h - tile:
        ys.append(h - tile)
    if w > tile and xs[-1] != w - tile:
        xs.append(w - tile)
    if h <= tile:
        ys = [0]
    if w <= tile:
        xs = [0]

    model.eval()
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                patch = cube[y0:y0 + tile, x0:x0 + tile, :]
                patch = _pad_if_needed(patch, tile, tile)
                x = torch.from_numpy(np.transpose(patch, (2, 0, 1))[None, ...]).float().to(device)
                probs = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                ph = min(tile, h - y0)
                pw = min(tile, w - x0)
                prob_acc[y0:y0 + ph, x0:x0 + pw] += probs[:ph, :pw]
                cnt_acc[y0:y0 + ph, x0:x0 + pw] += 1.0

    return prob_acc / np.maximum(cnt_acc, 1.0)


def predict(
    checkpoint: Path,
    input_hsi: Path,
    output_mask: Path,
    tile_size: int,
    overlap: int,
    threshold: float,
    drop_bands: Optional[int],
    num_bands: Optional[int],
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)

    in_channels = int(ckpt["in_channels"])
    model = UNetSmall(in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["model_state"])

    cube = _read_cube(input_hsi)

    ckpt_drop = int(ckpt.get("drop_bands", 0))
    ckpt_k = int(ckpt.get("num_bands", in_channels))
    use_drop = ckpt_drop if drop_bands is None else drop_bands
    use_k = ckpt_k if num_bands is None else num_bands

    if input_hsi.suffix.lower() != ".npy" or cube.shape[2] != in_channels:
        cube = _preprocess_cube(cube, drop_bands=use_drop, num_bands=use_k)

    if cube.shape[2] != in_channels:
        raise ValueError(f"Model expects {in_channels} channels but input has {cube.shape[2]} channels after preprocessing.")

    prob = _infer_tiled(model, cube, device=device, tile=tile_size, overlap=overlap)
    mask = (prob >= threshold).astype(np.uint8) * 255

    output_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_mask)
    np.save(output_mask.with_name(output_mask.stem + "_prob.npy"), prob)
    print(f"Saved mask: {output_mask}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/predict HSI gold-region segmentation model.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train model.")
    t.add_argument("--source", choices=["processed", "general"], default="general")
    t.add_argument("--processed-dir", type=Path, default=Path("HSIXRFDF") / "HSIXRFDF" / "processed")
    t.add_argument("--dataset-root", type=Path, default=Path("PCBDataset") / "PCBDataset")
    t.add_argument("--positive-class", type=int, default=2)
    t.add_argument("--drop-bands", type=int, default=10)
    t.add_argument("--num-bands", type=int, default=10)
    t.add_argument("--out-dir", type=Path, default=Path("HSIXRFDF") / "models")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--patch-size", type=int, default=128)
    t.add_argument("--train-samples-per-image", type=int, default=32)
    t.add_argument("--val-samples-per-image", type=int, default=12)
    t.add_argument("--val-ratio", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--num-workers", type=int, default=0)

    pred = sub.add_parser("predict", help="Predict mask for one HSI cube.")
    pred.add_argument("--checkpoint", type=Path, required=True)
    pred.add_argument("--input-hsi", type=Path, required=True)
    pred.add_argument("--output-mask", type=Path, required=True)
    pred.add_argument("--tile-size", type=int, default=256)
    pred.add_argument("--overlap", type=int, default=64)
    pred.add_argument("--threshold", type=float, default=0.5)
    pred.add_argument("--drop-bands", type=int, default=None)
    pred.add_argument("--num-bands", type=int, default=None)

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "train":
        cfg = TrainConfig(
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patch_size=args.patch_size,
            train_samples_per_image=args.train_samples_per_image,
            val_samples_per_image=args.val_samples_per_image,
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
            source=args.source,
            processed_dir=args.processed_dir,
            dataset_root=args.dataset_root,
            positive_class=args.positive_class,
            drop_bands=args.drop_bands,
            num_bands=args.num_bands,
        )
        train(cfg)
        return

    if args.cmd == "predict":
        predict(
            checkpoint=args.checkpoint,
            input_hsi=args.input_hsi,
            output_mask=args.output_mask,
            tile_size=args.tile_size,
            overlap=args.overlap,
            threshold=args.threshold,
            drop_bands=args.drop_bands,
            num_bands=args.num_bands,
        )
        return

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
