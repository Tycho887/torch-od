## Loading and Propagating TLEs

This library provides a functional, fully vectorized implementation of the SGP4 propagator written in native PyTorch. Unlike traditional object-oriented SGP4 libraries, this approach decodes Two-Line Elements (TLEs) directly into 1D or 2D parameter tensors. This allows you to leverage PyTorch's native broadcasting to propagate thousands of satellites across multiple timesteps simultaneously, completely avoiding Python loop overhead.

---

### Single TLE Propagation

To propagate a single satellite, use `tle_decode` to parse the string into a 1D tensor of orbital parameters, and pass those parameters into the `sgp4_propagate` function alongside your time tensor.

```python
import torch
from diffod.tle import tle_decode
from diffod.functional.sgp4 import sgp4_propagate

# 1. Define the TLE (e.g., ISS Zarya)
tle_str = [
    "1 25544U 98067A   20316.40015046  .00001878  00000-0  44436-4 0  9997",
    "2 25544  51.6465 289.4354 0001961 270.2184  89.8601 15.49504104255152"
]

# 2. Decode into a 1D parameter tensor
params = tle_decode(tle_str)

# 3. Define the propagation timeframe (e.g., 1000 minutes)
tsinces = torch.linspace(0, 1000, 1000, dtype=torch.float64)

# 4. Propagate
pos, vel = sgp4_propagate(
    tsinces,
    bstar=params[0], ndot=params[1], nddot=params[2],
    ecco=params[3], argpo=params[4], inclo=params[5],
    mo=params[6], no_kozai=params[7], nodeo=params[8]
)

# pos and vel shapes: (1000, 3)

```

---

### Batch TLE Propagation

For batch processing, `batch_decode` parses a list of TLEs into an `(N, 9)` tensor. By using `.unsqueeze(1)` on the sliced parameters, PyTorch automatically broadcasts the `(N, 1)` parameters against the `(T,)` time array, yielding a full `(N, T, 3)` state grid in a single vectorized pass.

```python
import torch
from diffod.tle import batch_decode
from diffod.functional.sgp4 import sgp4_propagate

# 1. Define multiple TLEs
tle_iss = [
    "1 25544U 98067A   20316.40015046  .00001878  00000-0  44436-4 0  9997",
    "2 25544  51.6465 289.4354 0001961 270.2184  89.8601 15.49504104255152"
]
tle_debris = [
    "1 43013U 17071A   21303.41500000  .00000112  00000-0  11234-4 0  9993",
    "2 43013  97.1234 100.5555 0011234 150.1234 210.9876 15.12345678123456"
]

# 2. Decode into a 2D batch tensor (Shape: N x 9)
batch_params = batch_decode([tle_iss, tle_debris])

# 3. Define the propagation timeframe
tsinces = torch.linspace(0, 1000, 1000, dtype=torch.float64)

# 4. Propagate using PyTorch broadcasting
pos, vel = sgp4_propagate(
    tsinces,
    bstar=batch_params[:, 0].unsqueeze(1),
    ndot=batch_params[:, 1].unsqueeze(1),
    nddot=batch_params[:, 2].unsqueeze(1),
    ecco=batch_params[:, 3].unsqueeze(1),
    argpo=batch_params[:, 4].unsqueeze(1),
    inclo=batch_params[:, 5].unsqueeze(1),
    mo=batch_params[:, 6].unsqueeze(1),
    no_kozai=batch_params[:, 7].unsqueeze(1),
    nodeo=batch_params[:, 8].unsqueeze(1)
)

# pos and vel shapes: (2, 1000, 3)

```

---

### Performance vs. dSGP4

Because our implementation leverages PyTorch's native JIT compilation and tensor broadcasting, it bypasses the heavy Python object-creation loops required by alternative libraries like `dSGP4` for batch processing.

The table below demonstrates the execution time (in seconds) of this custom propagator against `dSGP4`, highlighting the performance scaling across dense time grids and multi-satellite batches:

| Mode | Timesteps | Custom (s) | dSGP4 (s) | Speedup |
| --- | --- | --- | --- | --- |
| Single TLE | 1000 | 0.16236 | 0.15279 | 0.9x |
| Batch (2 TLEs) | 1000 | 0.00742 | 0.09008 | **12.1x** |
| Single TLE | 10000 | 0.00775 | 0.72517 | **93.6x** |
| Batch (2 TLEs) | 10000 | 0.06081 | 0.89647 | **14.7x** |
| Single TLE | 100000 | 0.33642 | 3.93730 | **11.7x** |
| Batch (2 TLEs) | 100000 | 0.78219 | 9.58732 | **12.3x** |

*(Note: The initial run for the 1000-step Single TLE includes the one-time overhead of the PyTorch JIT compiler warming up).*