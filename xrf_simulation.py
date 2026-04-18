"""Physically-informed XRF forward model for PCB metal mixtures.

This module upgrades a heuristic XRF simulation into a compact fundamental-parameter
style model with:
  - tube spectrum attenuation through layered samples (Beer-Lambert)
  - K-shell excitation gating using absorption edge energies
  - fluorescence yield / line branching
  - self-absorption of outgoing fluorescence photons
  - detector efficiency + Gaussian energy broadening
  - Poisson counting statistics

The implementation intentionally uses light-weight tabulated parameters and NumPy-only
numerics to keep it easy to run inside this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Lightweight elemental line data (keV). Values are representative and suitable
# for simulation/prototyping, not metrology-grade quantification.
ELEMENT_DB: Dict[str, Dict[str, float]] = {
    "Fe": {"z": 26, "k_edge": 7.112, "ka": 6.404, "kb": 7.058, "yield_k": 0.34},
    "Ni": {"z": 28, "k_edge": 8.333, "ka": 7.478, "kb": 8.265, "yield_k": 0.41},
    "Cu": {"z": 29, "k_edge": 8.979, "ka": 8.047, "kb": 8.905, "yield_k": 0.45},
    "Zn": {"z": 30, "k_edge": 9.659, "ka": 8.638, "kb": 9.572, "yield_k": 0.49},
    "Ag": {"z": 47, "k_edge": 25.514, "ka": 22.163, "kb": 24.942, "yield_k": 0.84},
    "Sn": {"z": 50, "k_edge": 29.200, "ka": 25.271, "kb": 28.486, "yield_k": 0.87},
    "Au": {"z": 79, "k_edge": 80.725, "ka": 68.804, "kb": 77.98, "yield_k": 0.96},
    "Pb": {"z": 82, "k_edge": 88.005, "ka": 74.970, "kb": 84.94, "yield_k": 0.97},
    "Pd": {"z": 46, "k_edge": 24.350, "ka": 21.177, "kb": 23.817, "yield_k": 0.83},
}


@dataclass(frozen=True)
class Layer:
    """Sample layer definition.

    thickness_cm: physical thickness in cm.
    density_g_cm3: bulk density.
    composition: mass fractions by element symbol (sums to ~1).
    """

    thickness_cm: float
    density_g_cm3: float
    composition: Mapping[str, float]


@dataclass(frozen=True)
class DetectorConfig:
    livetime_s: float = 10.0
    solid_angle_sr: float = 0.01
    area_cm2: float = 0.25
    fwhm_at_5_9keV: float = 0.16
    base_efficiency: float = 0.9
    low_e_rolloff_keV: float = 2.0


@dataclass(frozen=True)
class TubeConfig:
    voltage_kv: float = 50.0
    current_uA: float = 100.0
    takeoff_angle_deg: float = 45.0
    incidence_angle_deg: float = 45.0


def _validate_composition(composition: Mapping[str, float]) -> Dict[str, float]:
    missing = [el for el in composition if el not in ELEMENT_DB]
    if missing:
        raise ValueError(f"Unsupported elements in composition: {missing}")
    arr = np.array(list(composition.values()), dtype=float)
    if np.any(arr < 0):
        raise ValueError("Mass fractions must be non-negative")
    s = float(arr.sum())
    if s <= 0:
        raise ValueError("Composition must contain at least one positive fraction")
    return {k: float(v) / s for k, v in composition.items()}


def mass_attenuation_cm2_g(element: str, energy_keV: np.ndarray) -> np.ndarray:
    """Coarse mass attenuation model with K-edge jump behavior.

    Uses a power law with an edge multiplier. This is intentionally simple but
    captures the major shape required for forward simulation.
    """

    z = ELEMENT_DB[element]["z"]
    k_edge = ELEMENT_DB[element]["k_edge"]
    e = np.clip(np.asarray(energy_keV, dtype=float), 0.5, None)

    # Approximate photoelectric behavior ~ Z^n / E^m
    base = 0.014 * (z ** 3.8) / (e ** 3.1)

    # Approximate coherent+incoherent floor to avoid unrealistically tiny mu at high E
    scatter_floor = 0.02 + 0.0002 * z

    jump = np.where(e >= k_edge, 2.6, 0.9)
    return base * jump + scatter_floor


def mixture_mass_attenuation_cm2_g(composition: Mapping[str, float], energy_keV: np.ndarray) -> np.ndarray:
    comp = _validate_composition(composition)
    mu = np.zeros_like(np.asarray(energy_keV, dtype=float), dtype=float)
    for element, w in comp.items():
        mu += w * mass_attenuation_cm2_g(element, energy_keV)
    return mu


def tube_spectrum_bremsstrahlung(energy_grid_keV: np.ndarray, tube: TubeConfig) -> np.ndarray:
    """Generate relative tube spectrum (Kramers-like)."""

    e = np.asarray(energy_grid_keV, dtype=float)
    emax = tube.voltage_kv
    spectrum = np.where((e > 0.5) & (e < emax), (emax - e) * e, 0.0)

    # Add weak characteristic tube lines (assume Rh-like anode) for realism.
    for line_e, amp in ((20.2, 0.11), (22.7, 0.18)):
        spectrum += amp * np.exp(-0.5 * ((e - line_e) / 0.25) ** 2)

    # Scale with tube current (relative photons/s)
    return spectrum * tube.current_uA


def detector_efficiency(energy_keV: np.ndarray, det: DetectorConfig) -> np.ndarray:
    e = np.asarray(energy_keV, dtype=float)
    low_roll = 1.0 - np.exp(-np.clip(e, 0.0, None) / det.low_e_rolloff_keV)
    high_roll = np.exp(-0.0025 * np.clip(e - 22.0, 0.0, None))
    return np.clip(det.base_efficiency * low_roll * high_roll, 0.0, 1.0)


def gaussian_broaden(
    x_keV: np.ndarray,
    y: np.ndarray,
    fwhm_keV: float,
) -> np.ndarray:
    """Apply a constant-FWHM Gaussian convolution."""

    x = np.asarray(x_keV, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return y.copy()

    dx = float(np.mean(np.diff(x)))
    sigma = fwhm_keV / 2.355
    half_width = max(3, int(np.ceil(4.0 * sigma / dx)))
    kernel_x = np.arange(-half_width, half_width + 1) * dx
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def _escape_path_factor(layer: Layer, takeoff_angle_deg: float) -> float:
    theta = np.deg2rad(max(1.0, takeoff_angle_deg))
    return layer.thickness_cm / np.sin(theta)


def _incidence_path_factor(layer: Layer, incidence_angle_deg: float) -> float:
    theta = np.deg2rad(max(1.0, incidence_angle_deg))
    return layer.thickness_cm / np.sin(theta)


def _line_branching() -> Dict[str, Tuple[float, float]]:
    # (Kα, Kβ) branching fractions.
    return {el: (0.84, 0.16) for el in ELEMENT_DB}


def simulate_xrf_spectrum(
    layers: Sequence[Layer],
    energy_grid_keV: np.ndarray,
    tube: Optional[TubeConfig] = None,
    detector: Optional[DetectorConfig] = None,
    random_seed: Optional[int] = 7,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Simulate measured XRF spectrum counts for a multilayer sample.

    Returns:
      counts: Poisson-noisy detector counts on the supplied energy grid.
      line_truth: expected net line yields (pre-noise) for each element/line key.
    """

    tube = tube or TubeConfig()
    detector = detector or DetectorConfig()
    e = np.asarray(energy_grid_keV, dtype=float)
    if np.any(np.diff(e) <= 0):
        raise ValueError("energy_grid_keV must be strictly increasing")

    incident = tube_spectrum_bremsstrahlung(e, tube)
    line_yield_map: Dict[str, float] = {}
    spectrum = np.zeros_like(e, dtype=float)

    # Track transmitted tube flux through upper layers for deeper excitation.
    transmitted = incident.copy()

    for i, layer in enumerate(layers):
        comp = _validate_composition(layer.composition)
        mu_mix = mixture_mass_attenuation_cm2_g(comp, e)

        # Excitation attenuation in this layer.
        in_path = _incidence_path_factor(layer, tube.incidence_angle_deg)
        absorption = np.exp(-mu_mix * layer.density_g_cm3 * in_path)
        absorbed_flux = transmitted * (1.0 - absorption)

        # Fluorescence generation per element.
        for element, mass_frac in comp.items():
            props = ELEMENT_DB[element]
            k_edge = props["k_edge"]
            ka_e, kb_e = props["ka"], props["kb"]
            fy = props["yield_k"]

            excit_mask = e >= k_edge
            if not np.any(excit_mask):
                continue

            # Element-selective excitation weighting.
            sigma_el = mass_attenuation_cm2_g(element, e)
            sigma_mix = mu_mix + 1e-12
            excitation_frac = (mass_frac * sigma_el) / sigma_mix

            generated = absorbed_flux * excitation_frac * fy
            source_strength = float(np.trapezoid(generated[excit_mask], e[excit_mask]))
            if source_strength <= 0:
                continue

            br_ka, br_kb = _line_branching()[element]
            for line_name, line_e, br in (("Ka", ka_e, br_ka), ("Kb", kb_e, br_kb)):
                # Escape attenuation through this and overlying layers.
                escape = 1.0
                for over in layers[: i + 1]:
                    mu_out = mixture_mass_attenuation_cm2_g(over.composition, np.array([line_e]))[0]
                    out_path = _escape_path_factor(over, tube.takeoff_angle_deg)
                    escape *= np.exp(-mu_out * over.density_g_cm3 * out_path)

                geom = detector.solid_angle_sr / (4.0 * np.pi)
                det = detector_efficiency(np.array([line_e]), detector)[0]
                amp = source_strength * br * escape * geom * det

                key = f"{element}_{line_name}"
                line_yield_map[key] = line_yield_map.get(key, 0.0) + amp
                spectrum += amp * np.exp(-0.5 * ((e - line_e) / 0.05) ** 2)

        # Propagate transmitted tube flux to next layer.
        transmitted *= absorption

    # Add Compton/Rayleigh-like smooth background from transmitted flux.
    bkg = 0.008 * transmitted + 0.002 * incident
    spectrum += bkg

    # Detector resolution broadening and livetime scaling.
    broadened = gaussian_broaden(e, spectrum, detector.fwhm_at_5_9keV)
    expected_counts = np.clip(broadened * detector.livetime_s * detector.area_cm2, 0.0, None)

    rng = np.random.default_rng(random_seed)
    counts = rng.poisson(expected_counts)
    return counts.astype(float), line_yield_map


def estimate_element_scores(
    energy_grid_keV: np.ndarray,
    counts: np.ndarray,
    elements: Iterable[str],
    window_keV: float = 0.22,
) -> Dict[str, float]:
    """Return simple line-window scores useful for quick composition ranking."""

    e = np.asarray(energy_grid_keV, dtype=float)
    c = np.asarray(counts, dtype=float)
    scores: Dict[str, float] = {}

    for el in elements:
        if el not in ELEMENT_DB:
            continue
        ka = ELEMENT_DB[el]["ka"]
        kb = ELEMENT_DB[el]["kb"]

        ka_mask = np.abs(e - ka) <= window_keV
        kb_mask = np.abs(e - kb) <= window_keV
        side_mask = (np.abs(e - ka) > window_keV) & (np.abs(e - ka) <= 2 * window_keV)

        signal = c[ka_mask].sum() + 0.5 * c[kb_mask].sum()
        background = c[side_mask].sum() + 1e-9
        scores[el] = float(signal / background)

    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def demo() -> None:
    """Run a reproducible demonstration for a PCB-like multilayer stack."""

    energy = np.linspace(1.0, 40.0, 3901)
    layers = [
        Layer(
            thickness_cm=0.0012,
            density_g_cm3=8.3,
            composition={"Cu": 0.74, "Ni": 0.18, "Au": 0.08},
        ),
        Layer(
            thickness_cm=0.010,
            density_g_cm3=2.2,
            composition={"Fe": 0.05, "Cu": 0.20, "Zn": 0.35, "Sn": 0.40},
        ),
    ]

    counts, truth = simulate_xrf_spectrum(layers, energy)
    ranked = estimate_element_scores(energy, counts, ["Au", "Cu", "Ni", "Zn", "Sn", "Ag", "Pd"])

    print("Top inferred elements (score):")
    for el, score in list(ranked.items())[:5]:
        print(f"  {el}: {score:.3f}")

    print("\nStrongest simulated lines (expected amplitude):")
    for k, v in sorted(truth.items(), key=lambda kv: kv[1], reverse=True)[:8]:
        print(f"  {k}: {v:.6g}")


if __name__ == "__main__":
    demo()
