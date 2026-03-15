# Basic Usage

This guide covers loading TLEs, encoding/decoding them to tensors, and propagating single or multiple satellites.

## TLE Decoding and Encoding

- **Single TLE:** `tle_decode(tle_lines)` returns a 1‑D tensor `[bstar, ndot, nddot, ecco, argpo, inclo, mo, no_kozai, nodeo]`.  
  The inverse `tle_encode` reconstructs the two‑line format given the parameters, satellite number, and epoch.
- **Batch of TLEs:** `batch_decode(list_of_tles)` returns a 2‑D tensor of shape `(N, 9)`.  
  `batch_encode` reconstructs multiple TLE strings.

*Example (based on `test_tle.py`):*  
```python
from diffod.tle import tle_decode, tle_encode, batch_decode, batch_encode
import datetime

# Single TLE
params = tle_decode(tle_iss)
reconstructed_lines = tle_encode(
    *params.tolist(),          # unpack the 9 parameters
    sat_num=25544,
    epoch=datetime.datetime(2020, 11, 11, 9, 36, 13)  # example epoch
)
```

## Propagation

- **Single satellite:** `sgp4_propagate(tsinces, *params)` returns `(pos, vel)` each shaped `(T, 3)`.
- **Batch of satellites:** Unsqueeze the parameters to `(N, 1)` so they broadcast against the time vector `(T,)`.  
  The result shapes are `(N, T, 3)`.

```python
batch_params = batch_decode([tle_iss, tle_debris])          # (2, 9)
pos, vel = sgp4_propagate(
    tsinces,
    bstar=batch_params[:, 0].unsqueeze(1),
    ndot=batch_params[:, 1].unsqueeze(1),
    # ... all nine parameters
)
# pos.shape = (2, 1000, 3)
```

## Performance Considerations

- The propagator can be **JIT‑compiled** with `torch.jit.script` for even faster execution (see [Advanced Optimization](advanced_optimization.md)).
- The table below compares execution times against the legacy `dSGP4` library (from `benchmark.py`):

| Mode          | Timesteps | Custom (s) | dSGP4 (s) | Speedup |
|---------------|-----------|------------|-----------|---------|
| Single TLE    | 1000      | 0.16236    | 0.15279   | 0.9x    |
| Batch (2 TLEs)| 1000      | 0.00742    | 0.09008   | **12.1x** |
| …             | …         | …          | …         | …       |

*Note: The first run includes PyTorch JIT warm‑up overhead.*