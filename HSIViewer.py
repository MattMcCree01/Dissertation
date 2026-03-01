import os
from pathlib import Path
import spectral.io.envi as envi
from spectral import imshow
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the image using the header file
# Note: Ensure pcb1.hdr and the data file (e.g. pcb1.dat) are in the same folder
script_dir = Path(__file__).parent
hdr_path = script_dir / 'PCBDataset' / 'PCBDataset' / 'HSI' / 'pcb2' / 'pcb2.hdr'
if not hdr_path.exists():
    raise FileNotFoundError(f"HSI header not found: {hdr_path}")

img = envi.open(str(hdr_path))

# 2. Access wavelength data directly from metadata
# 2. Access wavelength data directly from metadata (robust fallback)
try:
    wavelengths = img.bands.centers
except Exception:
    md = getattr(img, 'metadata', None)
    if md is None:
        raise RuntimeError('Unable to read wavelength information from HSI file')
    # metadata 'wavelength' may be a string list
    wl = md.get('wavelength') or md.get('wavelengths') or md.get('Wavelength')
    if wl is None:
        raise RuntimeError('Wavelengths not found in HSI metadata')
    # ensure numeric numpy array
    wavelengths = np.array([float(w) for w in wl])

print(f"Detected {len(wavelengths)} bands.")
print(f"Wavelength range: {wavelengths[0]} nm to {wavelengths[-1]} nm")

# 3. Create an RGB view using the default bands from your header
# Band 89 (~614nm - Red), Band 64 (~571nm - Green), Band 28 (~511nm - Blue)
# We subtract 1 because Python uses 0-based indexing
rgb_bands = (88, 63, 27) 

print("Displaying RGB image...")
imshow(img, bands=rgb_bands, title="PCB Hyperspectral Image (True Color)")
plt.ioff()
plt.show()

# 4. Precious metal characteristic wavelengths (nm)
metal_wavelengths = {
    'Gold': [580, 620],      # Strong reflectance in yellow-red
    'Silver': [400, 700],    # High reflectance across visible spectrum
    'Copper': [600, 650],    # Red-orange reflectance
    'Palladium': [450, 550]  # Green-blue reflectance
}

# 5. Plot intensity at specific wavelengths for each metal
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Precious Metal Wavelength Highlights', fontsize=16)

data = img.load()
print(f"Raw data shape: {getattr(data, 'shape', None)}")
# Ensure data is a NumPy array
data = np.array(data)
# Normalize axes: prefer (rows, cols, bands)
if data.ndim == 3:
    if data.shape[2] == len(wavelengths):
        # Already in correct format (rows, cols, bands)
        pass
    elif data.shape[0] == len(wavelengths):
        # Format is (bands, rows, cols) - transpose to (rows, cols, bands)
        data = np.transpose(data, (1, 2, 0))
    elif data.shape[1] == len(wavelengths):
        # Format is (rows, bands, cols) - transpose to (rows, cols, bands)
        data = np.transpose(data, (0, 2, 1))
    else:
        # Mismatch - try squeezing and retry
        data = np.squeeze(data)
        if data.ndim == 3 and data.shape[0] == len(wavelengths):
            data = np.transpose(data, (1, 2, 0))

print(f"Normalized data shape: {data.shape}")

# Ensure wavelengths matches the number of bands in the data
if len(wavelengths) != data.shape[2]:
    print(f"Warning: wavelengths ({len(wavelengths)}) != data bands ({data.shape[2]}). Trimming wavelengths.")
    wavelengths = np.array(wavelengths[:data.shape[2]])
else:
    wavelengths = np.array(wavelengths)

metals = list(metal_wavelengths.keys())

for idx, (metal, wl_range) in enumerate(metal_wavelengths.items()):
    ax = axes[idx // 2, idx % 2]
    
    # Find band closest to the characteristic wavelength
    target_wl = np.mean(wl_range)
    band_idx = np.argmin(np.abs(np.array(wavelengths) - target_wl))
    
    im = ax.imshow(data[:, :, band_idx], cmap='hot')
    ax.set_title(f'{metal} (~{wavelengths[band_idx]:.1f} nm)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.show()


# Pick a pixel coordinate (y, x) from your imshow() view
# e.g., one of the gold connector pins at the bottom
pixel_y, pixel_x = 201, 256
sample_spectrum = data[pixel_y, pixel_x, :]
# Ensure 1D spectrum for plotting
sample_spectrum = np.asarray(sample_spectrum).squeeze()
if sample_spectrum.ndim != 1:
    sample_spectrum = sample_spectrum.ravel()

plt.figure(figsize=(8, 4))
plt.plot(wavelengths, sample_spectrum)
plt.title(f"Spectral Profile at ({pixel_y}, {pixel_x})")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.grid(True)
plt.show()
def get_gold_mask(data, wavelengths, connector_mask=None, red_pct=95, edge_pct=75):
    """Compute red/blue ratio and red-green edge strength, choose thresholds
    using percentiles (optionally restricted to connector_mask), and return
    mask plus diagnostic maps and thresholds.
    Returns: mask, red_index, edge_strength, red_thresh, edge_thresh
    """
    # Find band indices for gold-specific wavelengths
    idx_450 = np.argmin(np.abs(wavelengths - 450))  # Blue
    idx_550 = np.argmin(np.abs(wavelengths - 550))  # Green
    idx_600 = np.argmin(np.abs(wavelengths - 600))  # Red-yellow

    # Extract bands (float)
    blue = data[:, :, idx_450].astype(float)
    green = data[:, :, idx_550].astype(float)
    yellow_red = data[:, :, idx_600].astype(float)

    # Diagnostic indices
    red_index = yellow_red / (blue + 1e-6)  # ratio of red to blue
    edge_strength = (yellow_red - green) / (yellow_red + green + 1e-6)  # red-green edge

    # Select values used to compute thresholds (prefer connector region if provided)
    if connector_mask is not None and connector_mask.any():
        sample_red = red_index[connector_mask]
        sample_edge = edge_strength[connector_mask]
    else:
        sample_red = red_index.ravel()
        sample_edge = edge_strength.ravel()

    # Avoid empty samples
    if sample_red.size == 0:
        sample_red = red_index.ravel()
    if sample_edge.size == 0:
        sample_edge = edge_strength.ravel()

    # Percentile-based automatic thresholds
    red_thresh = float(np.percentile(sample_red, red_pct))
    edge_thresh = float(np.percentile(sample_edge, edge_pct))

    # Final mask: require both a high red/blue ratio and a strong red-green edge
    mask = (red_index >= red_thresh) & (edge_strength >= edge_thresh)

    return mask, red_index, edge_strength, red_thresh, edge_thresh

# Load general segmentation mask and isolate connector regions (class 3)
gm_path = script_dir / 'PCBDataset' / 'PCBDataset' / 'HSI' / 'General_masks' / '1'
if gm_path.exists():
    gmask_data = np.fromfile(str(gm_path), dtype=np.uint8)
    try:
        gmask = gmask_data.reshape(data.shape[0], data.shape[1])
    except ValueError:
        # try the transposed shape if needed
        try:
            gmask = gmask_data.reshape(data.shape[1], data.shape[0]).T
        except Exception:
            raise
    connector_region = (gmask == 3)
    print(f"Loaded general mask from {gm_path} — connector pixels: {connector_region.sum()}")
else:
    connector_region = np.ones((data.shape[0], data.shape[1]), dtype=bool)
    print("General mask not found — running detection on full image")

# Run gold detection using percentiles computed inside connector regions
mask_all, red_index, edge_strength, red_thr, edge_thr = get_gold_mask(
    data, wavelengths, connector_mask=connector_region, red_pct=95, edge_pct=75
)

# Restrict to connectors
mask = mask_all & connector_region

# Create RGB view from the data
rgb_view = data[:, :, rgb_bands].astype(np.float32)
rgb_view = (rgb_view - rgb_view.min(axis=(0,1))) / (rgb_view.max(axis=(0,1)) - rgb_view.min(axis=(0,1)) + 1e-6)

plt.figure(figsize=(18, 5))
plt.subplot(1, 4, 1)
plt.title(f"Red/Blue Ratio (thr={red_thr:.2f})")
plt.imshow(red_index, cmap='viridis')
plt.colorbar()

plt.subplot(1, 4, 2)
plt.title(f"Red-Green Edge (thr={edge_thr:.2f})")
plt.imshow(edge_strength, cmap='YlOrBr')
plt.colorbar()

plt.subplot(1, 4, 3)
plt.title("Connector Regions (class 3)")
plt.imshow(connector_region, cmap='gray')
plt.colorbar()

plt.subplot(1, 4, 4)
plt.title("Gold in Connectors (final)")
plt.imshow(rgb_view)
plt.imshow(mask, alpha=0.6, cmap='jet')
plt.colorbar()

plt.tight_layout()
plt.show()

print(f"Thresholds used: red_index >= {red_thr:.3f}, edge_strength >= {edge_thr:.3f}")
print(f"Gold candidate pixels (within connectors): {np.count_nonzero(mask)}")