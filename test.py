import torch
import time
import numpy as np
from dsgp4.tle import TLE
import dsgp4
from diffod.functional.sgp4 import sgp4_propagate

# Assume these are your new functional implementations
# from diffod.functional.sgp4 import sgp4_propagate, GravConsts
# For this script to be runnable, I will assume sgp4_propagate is available 
# (either imported or defined as in previous steps)

# ---------------------------------------------------------
# 0. Helper: Robust Timing Function
# ---------------------------------------------------------
def benchmark_func(name, func, *args, n_iters=5, warmup=False, **kwargs):
    """Runs a function multiple times and reports average time."""
    
    # Warmup (important for CUDA and compilation)
    if warmup:
        print(f"  [Warmup] {name}...")
        for _ in range(3):
            _ = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Timing
    times = []
    print(f"  [Timing] {name} ({n_iters} runs)...")
    for _ in range(n_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        
        # Run function
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)
    
    avg_time = np.mean(times)
    print(f"  -> Average Time: {avg_time:.6f} s")
    return result, avg_time

# ---------------------------------------------------------
# 1. Setup Data
# ---------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device.upper()}")

# Load TLE
TLE_list = [
    "ISS (ZARYA)",
    "1 25544U 98067A   26038.50283897  .00012054  00000-0  23050-3 0  9996",
    "2 25544  51.6315 221.5822 0011000  74.6214 285.5989 15.48462076551652",
]
# Initialize standard TLE object
init_tle = TLE(data=TLE_list)

# Move TLE data to device if necessary
# (Assuming TLE object attributes are already tensors, we define wrappers below)

# Generate massive time batch (1 million points)
tsince = torch.linspace(0, 10000, 1_000_000, dtype=torch.float64, device=device)

# Define Constants (WGS-84 usually) for Functional Version
# You might need to adjust these values to match exactly what `dsgp4` uses internally
class GravConsts:
    tumin = 1.0 / 13.446839
    mu = 398600.8
    radiusearthkm = 6378.135
    xke = 0.0743669161
    j2 = 0.001082616
    j3 = -0.00000253881
    j4 = -0.00000165597
    j3oj2 = j3 / j2

consts = GravConsts()

# ---------------------------------------------------------
# 2. Define Wrappers for Benchmarking
# ---------------------------------------------------------

# Wrapper A: Reference (Original dsgp4)
def run_reference():
    # dsgp4.propagate handles initialization internally if initialized=False
    # We move tsince to cpu if dsgp4 doesn't support gpu, or keep it if it does.
    return dsgp4.propagate(tle=init_tle, tsinces=tsince, initialized=False)

# Wrapper B: Functional (Eager)
def run_functional():
    # Ensure inputs are on the correct device
    return sgp4_propagate(
        consts=consts,
        tsince=tsince,
        bstar=init_tle._bstar.to(device),
        ndot=init_tle._ndot.to(device),
        nddot=init_tle._nddot.to(device),
        ecco=init_tle._ecco.to(device),
        argpo=init_tle._argpo.to(device),
        inclo=init_tle._inclo.to(device),
        mo=init_tle._mo.to(device),
        no_kozai=init_tle._no_kozai.to(device),
        nodeo=init_tle._nodeo.to(device)
    )

# Wrapper C: Functional (Compiled)
# We compile the function once. 
# mode="reduce-overhead" is often good for many small calls, 
# "max-autotune" is best for massive throughput but takes longer to compile.
compiled_sgp4 = torch.compile(sgp4_propagate, mode="reduce-overhead")

def run_compiled():
    return compiled_sgp4(
        consts=consts,
        tsince=tsince,
        bstar=init_tle._bstar.to(device),
        ndot=init_tle._ndot.to(device),
        nddot=init_tle._nddot.to(device),
        ecco=init_tle._ecco.to(device),
        argpo=init_tle._argpo.to(device),
        inclo=init_tle._inclo.to(device),
        mo=init_tle._mo.to(device),
        no_kozai=init_tle._no_kozai.to(device),
        nodeo=init_tle._nodeo.to(device)
    )

# ---------------------------------------------------------
# 3. Run Benchmarks
# ---------------------------------------------------------
print("\n--- Starting Benchmark ---")

# 1. Reference
ref_res, ref_time = benchmark_func("Reference (dSGP4)", run_reference, n_iters=5, warmup=True)
r_ref, v_ref = ref_res[:, 0], ref_res[:, 1]

# 2. Functional Eager
func_res, func_time = benchmark_func("Functional (Eager)", run_functional, n_iters=5, warmup=True)
r_func, v_func = func_res

# 3. Functional Compiled
# Note: First run will be slow due to compilation! 'warmup=True' handles this.
comp_res, comp_time = benchmark_func("Functional (Compiled)", run_compiled, n_iters=5, warmup=True)
r_comp, v_comp = comp_res

# ---------------------------------------------------------
# 4. Analysis
# ---------------------------------------------------------
print("\n--- Results Summary ---")
print(f"Reference Time: {ref_time:.4f} s")
print(f"Functional Time: {func_time:.4f} s (Speedup: {ref_time/func_time:.2f}x)")
print(f"Compiled Time:   {comp_time:.4f} s (Speedup: {ref_time/comp_time:.2f}x)")

# ---------------------------------------------------------
# 5. Correctness Check
# ---------------------------------------------------------
print("\n--- Correctness Check (Normalized Error) ---")

# Move ref to same device for comparison
r_ref = r_ref.to(device)
v_ref = v_ref.to(device)

def check_error(name, r, v):
    err_r = torch.linalg.norm(r - r_ref) / torch.linalg.norm(r_ref)
    err_v = torch.linalg.norm(v - v_ref) / torch.linalg.norm(v_ref)
    print(f"{name}:")
    print(f"  Pos Error: {err_r:.2e}")
    print(f"  Vel Error: {err_v:.2e}")
    if err_r > 1e-5:
        print("  WARNING: High position error detected.")

check_error("Functional Eager", r_func, v_func)
check_error("Functional Compiled", r_comp, v_comp)