import pytest
import torch
import requests
import dsgp4
from dsgp4.tle import TLE
from diffod.functional.sgp4 import sgp4_propagate

# ---------------------------------------------------------
# 1. Fixture: Load and Filter TLEs
# ---------------------------------------------------------
@pytest.fixture(scope="module")
def leo_tles() -> list:
    """Fetches 100 LEO TLEs from CelesTrak."""
    print("\nFetching TLEs from CelesTrak...")
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.text.splitlines()
    except Exception as e:
        pytest.fail(f"Failed to download TLEs: {e}")

    # Parse and filter for LEO (Mean Motion > 11.25 roughly corresponds to period < 128 min)
    # This ensures we test the "Near Earth" SGP4 path primarily.
    filtered_tles = []
    
    # TLE format: Line 1, Line 2. We need pairs.
    # We iterate by 3 because CelesTrak usually gives: Name, Line1, Line2
    for i in range(0, len(data), 3):
        if i+2 >= len(data): break
        name = data[i].strip()
        line1 = data[i+1].strip()
        line2 = data[i+2].strip()
        
        # Simple check for Mean Motion (Columns 53-63 in Line 2)
        try:
            mean_motion = float(line2[52:63])
            if mean_motion > 11.25: # LEO filter
                filtered_tles.append(name)
                filtered_tles.append(line1)
                filtered_tles.append(line2)
        except ValueError:
            continue
            
        if len(filtered_tles) >= 300: # 100 satellites * 3 lines
            break
            
    return filtered_tles

# ---------------------------------------------------------
# 2. Fixture: Setup Tensors
# ---------------------------------------------------------
@pytest.fixture(scope="module")
def setup_data(leo_tles: list) -> tuple[TLE, torch.Tensor]:
    """Prepares the TLE objects and tensors."""
    # Create dsgp4 TLE object
    tle_obj = TLE(data=leo_tles)
    
    # Create time batch (1 day of propagation, 1-minute steps)
    # 1440 minutes
    tsince = torch.linspace(0, 1440, 1440, dtype=torch.float64)
    
    # Create Constants
    
    return tle_obj, tsince

# ---------------------------------------------------------
# 3. Helper: Comparison Logic
# ---------------------------------------------------------
def assert_close(r1, v1, r2, v2, tol_pos=1e-5, tol_vel=1e-6) -> None:
    """
    Asserts that position and velocity match within tolerance.
    Units: km and km/s
    """
    # Relative Error
    diff_r = torch.linalg.norm(r1 - r2, dim=-1)
    diff_v = torch.linalg.norm(v1 - v2, dim=-1)
    
    norm_r = torch.linalg.norm(r2, dim=-1)
    norm_v = torch.linalg.norm(v2, dim=-1)

    max_err_r = (diff_r / norm_r).max().item()
    max_err_v = (diff_v / norm_v).max().item()
    
    print(f"    Max Rel Pos Error: {max_err_r:.2e}")
    print(f"    Max Rel Vel Error: {max_err_v:.2e}")

    assert max_err_r < tol_pos, f"Position error too high: {max_err_r}"
    assert max_err_v < tol_vel, f"Velocity error too high: {max_err_v}"

# ---------------------------------------------------------
# 4. Tests
# ---------------------------------------------------------

def test_sgp4_eager(setup_data: tuple[TLE, torch.Tensor]) :
    """Compares Reference vs Functional (Eager Mode)"""
    tle, tsince = setup_data
    
    print(f"\n[Test] Eager Mode (Satellites: {len(tle._bstar)})")

    # 1. Run Reference
    # dsgp4.propagate returns shape (N_sats, N_times, 2, 3) usually, or (N_sats*N_times, ...)
    # Let's check the shape. The standard `propagate` usually flattens or batches.
    # We assume standard behavior: returns [Batch, 2, 3] where Batch = N_sats * N_times
    # OR [N_sats, N_times, 2, 3] depending on version.
    
    # We will force the reference to run per satellite to ensure clarity or use broadcating if supported.
    # dsgp4.propagate supports broadcasting tsinces against TLEs.
    
    # Let's align dimensions:
    # TLEs: [N]
    # Time: [M]
    # Result expected: [N, M, 2, 3]
    
    # Run Reference
    # Note: dsgp4 implementation might vary. 
    # If it returns [N, 6] for scalar time, or [N, M, 6]...
    # We will use the functional code's input shape logic for the test.
    
    # Expand inputs for broadcasting manually to be safe for functional input
    # TLE params: [N] -> [N, 1] -> broadcast to [N, M] -> flatten to [N*M]
    n_sats = len(tle._bstar)
    n_times = len(tsince)
    
    # Prepare inputs for Functional
    bstar = tle._bstar.unsqueeze(1).expand(-1, n_times).reshape(-1)
    ndot = tle._ndot.unsqueeze(1).expand(-1, n_times).reshape(-1)
    nddot = tle._nddot.unsqueeze(1).expand(-1, n_times).reshape(-1)
    ecco = tle._ecco.unsqueeze(1).expand(-1, n_times).reshape(-1)
    argpo = tle._argpo.unsqueeze(1).expand(-1, n_times).reshape(-1)
    inclo = tle._inclo.unsqueeze(1).expand(-1, n_times).reshape(-1)
    mo = tle._mo.unsqueeze(1).expand(-1, n_times).reshape(-1)
    no_kozai = tle._no_kozai.unsqueeze(1).expand(-1, n_times).reshape(-1)
    nodeo = tle._nodeo.unsqueeze(1).expand(-1, n_times).reshape(-1)
    
    tsince_flat = tsince.unsqueeze(0).expand(n_sats, -1).reshape(-1)

    # Run Functional (Eager)
    r_func, v_func = sgp4_propagate(
        tsince_flat, bstar, ndot, nddot, ecco, argpo, inclo, mo, no_kozai, nodeo
    )
    
    # Run Reference
    # dsgp4.propagate(tle, tsinces) usually handles the broadcast.
    # It returns [N_sats, N_times, 6] or similar.
    ref_out = dsgp4.propagate(tle, tsince, initialized=False)
    
    # Reshape reference to match flat functional [N*M, 3]
    # Reference out is likely [N_sats, N_times, 2, 3] or [N_sats, N_times, 6]
    # Checking shape... usually [Batch, 2, 3] or [Batch, 6]
    # For this test, we reshape ref_out to flat [N*M, 3]
    
    # Assuming ref_out is [N_sats, N_times, 2, 3] -> [N*M, 2, 3]
    # If ref_out is [N_sats * N_times, 2, 3], we are good.
    if ref_out.ndim == 4:
         ref_out = ref_out.reshape(-1, 2, 3)
         
    r_ref = ref_out[:, 0, :]
    v_ref = ref_out[:, 1, :]
    
    assert_close(r_func, v_func, r_ref, v_ref)


def test_sgp4_compiled(setup_data: tuple[TLE, torch.Tensor]) -> None:
    """Compares Reference vs Functional (Compiled Mode)"""
    tle, tsince = setup_data
    
    # Only run if torch.compile is supported (Linux/Mac, PyTorch 2.0+)
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")

    print("\n[Test] Compiled Mode")

    # Prepare Data (Flattened)
    n_sats = len(tle._bstar)
    n_times = len(tsince)
    
    bstar = tle._bstar.unsqueeze(1).expand(-1, n_times).reshape(-1)
    ndot = tle._ndot.unsqueeze(1).expand(-1, n_times).reshape(-1)
    nddot = tle._nddot.unsqueeze(1).expand(-1, n_times).reshape(-1)
    ecco = tle._ecco.unsqueeze(1).expand(-1, n_times).reshape(-1)
    argpo = tle._argpo.unsqueeze(1).expand(-1, n_times).reshape(-1)
    inclo = tle._inclo.unsqueeze(1).expand(-1, n_times).reshape(-1)
    mo = tle._mo.unsqueeze(1).expand(-1, n_times).reshape(-1)
    no_kozai = tle._no_kozai.unsqueeze(1).expand(-1, n_times).reshape(-1)
    nodeo = tle._nodeo.unsqueeze(1).expand(-1, n_times).reshape(-1)
    tsince_flat = tsince.unsqueeze(0).expand(n_sats, -1).reshape(-1)

    # Compile
    # fullgraph=True ensures no python fallback (validates our functional logic is sound)
    compiled_func = torch.compile(sgp4_propagate, fullgraph=True)
    
    # Run Compiled
    r_comp, v_comp = compiled_func(
        tsince_flat, bstar, ndot, nddot, ecco, argpo, inclo, mo, no_kozai, nodeo
    )
    
    # Run Reference
    ref_out = dsgp4.propagate(tle, tsince, initialized=False)
    if ref_out.ndim == 4:
         ref_out = ref_out.reshape(-1, 2, 3)
         
    r_ref = ref_out[:, 0, :]
    v_ref = ref_out[:, 1, :]

    assert_close(r_comp, v_comp, r_ref, v_ref)

if __name__ == "__main__":
    # Allows running script directly without pytest command
    pytest.main([__file__])