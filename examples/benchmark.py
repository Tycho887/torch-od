import time

import dsgp4
import torch

from torch_sgp4.propagators.sgp4 import sgp4_propagate
from torch_sgp4.tle import batch_decode

compiled_propagate = torch.jit.script(sgp4_propagate)

# --- Test Data ---
TLE_STRINGS = [
    [
        "1 25544U 98067A   20316.40015046  .00001878  00000-0  44436-4 0  9997",
        "2 25544  51.6465 289.4354 0001961 270.2184  89.8601 15.49504104255152",
    ],
    [
        "1 43013U 17071A   21303.41500000  .00000112  00000-0  11234-4 0  9993",
        "2 43013  97.1234 100.5555 0011234 150.1234 210.9876 15.12345678123456",
    ],
]


def benchmark_custom(tensor_batch, tsinces):
    """Benchmarks the PyTorch broadcasted propagator."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    pos, vel = compiled_propagate(
        tsinces,
        bstar=tensor_batch[:, 0].unsqueeze(1),
        ndot=tensor_batch[:, 1].unsqueeze(1),
        nddot=tensor_batch[:, 2].unsqueeze(1),
        ecco=tensor_batch[:, 3].unsqueeze(1),
        argpo=tensor_batch[:, 4].unsqueeze(1),
        inclo=tensor_batch[:, 5].unsqueeze(1),
        mo=tensor_batch[:, 6].unsqueeze(1),
        no_kozai=tensor_batch[:, 7].unsqueeze(1),
        nodeo=tensor_batch[:, 8].unsqueeze(1),
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.perf_counter() - t0


def benchmark_dsgp4(tles_strs, tsinces):
    """Benchmarks the legacy dSGP4 propagator."""
    num_steps = len(tsinces)
    num_tles = len(tles_strs)

    # Note: We include setup time here because the legacy API forces this
    # highly inefficient list generation just to perform a batch operation.
    t0 = time.perf_counter()
    tles_obj = [dsgp4.tle.TLE(tle) for tle in tles_strs]
    tles_flat = []
    for tle in tles_obj:
        tles_flat += [tle] * num_steps

    tsinces_flat = torch.cat([tsinces] * num_tles)
    _, tle_batch = dsgp4.initialize_tle(tles_flat)

    _ = dsgp4.propagate_batch(tle_batch, tsinces_flat)

    return time.perf_counter() - t0


def run_benchmarks():
    torch.set_default_dtype(torch.float64)
    timesteps_to_test = [1_000, 10_000, 100_000]

    # Pre-decode parameters for custom propagator
    single_tensor = batch_decode([TLE_STRINGS[0]])
    batch_tensor = batch_decode(TLE_STRINGS)

    # Warmup the JIT compiler to ensure fair timing
    _ = benchmark_custom(batch_tensor, torch.linspace(0, 10, 10))

    print(
        f"{'Mode':<15} | {'Timesteps':<10} | {'Custom (s)':<12} | {'dSGP4 (s)':<12} | {'Speedup':<10}"
    )
    print("-" * 70)

    for steps in timesteps_to_test:
        tsinces = torch.linspace(0, 1440, steps)

        # --- Single TLE Benchmark ---
        custom_time = benchmark_custom(single_tensor, tsinces)
        dsgp4_time = benchmark_dsgp4([TLE_STRINGS[0]], tsinces)
        speedup = dsgp4_time / custom_time
        print(
            f"{'Single TLE':<15} | {steps:<10} | {custom_time:<12.5f} | {dsgp4_time:<12.5f} | {speedup:.1f}x"
        )

        # --- Batch TLE Benchmark ---
        custom_time_batch = benchmark_custom(batch_tensor, tsinces)

        # Guardrail: Prevent dSGP4 from crashing the system on 1M steps for batches
        # if steps == 1_000_000:
        #     dsgp4_time_batch = float('inf')
        #     dsgp4_str = "OOM/Skip"
        #     speedup_str = "N/A"
        # else:
        dsgp4_time_batch = benchmark_dsgp4(TLE_STRINGS, tsinces)
        dsgp4_str = f"{dsgp4_time_batch:.5f}"
        speedup_str = f"{dsgp4_time_batch / custom_time_batch:.1f}x"

        print(
            f"{'Batch (2 TLEs)':<15} | {steps:<10} | {custom_time_batch:<12.5f} | {dsgp4_str:<12} | {speedup_str:<10}"
        )


if __name__ == "__main__":
    run_benchmarks()
