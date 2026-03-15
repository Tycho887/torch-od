# DiffOD: Differentiable Orbit Determination

DiffOD is a PyTorch-based library for differentiable SGP4 propagation and orbit determination.  
It provides a **functional, fully vectorized** implementation of the SGP4 propagator that leverages PyTorch’s broadcasting and autograd, enabling:

- Efficient batch processing of thousands of satellites across many time steps.
- Gradients of satellite states with respect to TLE parameters.
- Custom objective functions and optimizers for orbit determination.

## Installation

```bash
pip install diffod
```

Or install in development mode from source:

```bash
git clone https://github.com/Tycho887/diffod.git
cd diffod
pip install -e .
```

## Overview

Traditional SGP4 libraries are object‑oriented and loop over satellites and time, which is slow and incompatible with automatic differentiation.  
DiffOD instead decodes Two‑Line Elements (TLEs) into parameter tensors and propagates them using vectorized operations.  
This design allows you to:

- Propagate **N** satellites for **T** time steps in a single batched call.
- Compute gradients of positions/velocities w.r.t. the nine TLE parameters.
- Build differentiable measurement models (Doppler, range, biases) and solve orbit determination problems with gradient‑based optimizers or a dedicated Gauss‑Newton solver.

## Getting Started

Here is a minimal example to load a TLE and propagate it:

```python
import torch
from diffod.tle import tle_decode
from diffod.functional.sgp4 import sgp4_propagate

# TLE for the ISS (Zarya)
tle_lines = [
    "1 25544U 98067A   20316.40015046  .00001878  00000-0  44436-4 0  9997",
    "2 25544  51.6465 289.4354 0001961 270.2184  89.8601 15.49504104255152"
]

# Decode into a 1‑D tensor of 9 parameters
params = tle_decode(tle_lines)

# Define time steps (minutes from epoch)
tsinces = torch.linspace(0, 1440, 1000)

# Propagate (returns positions and velocities, each of shape (1000, 3))
pos, vel = sgp4_propagate(tsinces, *params.unbind())

print(pos)
```

For more detailed examples, please refer to the User Guide:

- [Basic Usage](docs/basic_usage.md) – TLE encoding/decoding, single and batch propagation, performance comparisons.
- [Advanced Features](docs/advanced_features.md) – Observation models (Doppler, range, biases) and the Gauss‑Newton SVD solver.
- [Advanced Optimization](docs/advanced_optimization.md) – JIT compilation, automatic differentiation, and using PyTorch optimizers (LBFGS, Adam) for custom orbit determination.

## Citation

If you use DiffOD in your research, please cite …

## License