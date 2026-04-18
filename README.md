# Dissertation

This repository contains exploratory code and notes for hyperspectral PCB analysis.

## Improved XRF simulation

The previous XRF logic was mainly heuristic. A new physically-informed forward model is
now available in `xrf_simulation.py` with these upgrades:

- Multilayer sample support (`Layer` dataclass).
- Incident tube spectrum generation (Kramers-style bremsstrahlung + weak tube lines).
- Beer-Lambert attenuation on both incident and emitted paths.
- K-shell edge gating, fluorescence yield, and Kα/Kβ branching.
- Detector response: efficiency roll-off + Gaussian broadening + Poisson noise.
- Quick post-simulation element ranking via line-window scores.

### Run the demonstration

```bash
python xrf_simulation.py
```

### Use from another script

```python
import numpy as np
from xrf_simulation import Layer, simulate_xrf_spectrum, estimate_element_scores

energy = np.linspace(1.0, 40.0, 3901)
layers = [
    Layer(thickness_cm=0.001, density_g_cm3=8.9, composition={"Cu": 0.9, "Ni": 0.1}),
    Layer(thickness_cm=0.010, density_g_cm3=2.5, composition={"Zn": 0.6, "Sn": 0.4}),
]

counts, truth = simulate_xrf_spectrum(layers, energy)
ranked = estimate_element_scores(energy, counts, ["Cu", "Ni", "Zn", "Sn", "Au"])
print(ranked)
```

## Existing HSI viewer

`HSIViewer.py` remains available for hyperspectral visualization and heuristic connector/gold
candidate highlighting.
