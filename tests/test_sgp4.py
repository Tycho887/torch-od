import dsgp4
import pytest
import torch

from torch_sgp4.propagators.sgp4 import sgp4_propagate
from torch_sgp4.tle import batch_decode, tle_decode

# JIT Compile the custom propagator for the tests
compiled_propagate = torch.jit.script(sgp4_propagate)

# --- Test Data ---
TLE_ISS = [
    "1 25544U 98067A   20316.40015046  .00001878  00000-0  44436-4 0  9997",
    "2 25544  51.6465 289.4354 0001961 270.2184  89.8601 15.49504104255152",
]
TLE_DEBRIS = [
    "1 43013U 17071A   21303.41500000  .00000112  00000-0  11234-4 0  9993",
    "2 43013  97.1234 100.5555 0011234 150.1234 210.9876 15.12345678123456",
]


@pytest.fixture(autouse=True)
def set_default_dtype():
    """Ensure float64 is used for all SGP4 operations to prevent truncation."""
    torch.set_default_dtype(torch.float64)


@pytest.fixture
def tsinces():
    """Standard 24-hour propagation array (1000 steps)."""
    return torch.linspace(0, 1440, 1000)


@pytest.mark.parametrize("tle_str", [TLE_ISS, TLE_DEBRIS], ids=["ISS", "Debris"])
def test_single_tle_propagation(tle_str, tsinces):
    """Test single TLE propagation accuracy against dSGP4."""
    # --- dSGP4 Baseline ---
    dtle = dsgp4.tle.TLE(tle_str)
    dsgp4.initialize_tle(dtle)
    state_dsgp4 = dsgp4.propagate(dtle, tsinces)
    pos_baseline = state_dsgp4[:, 0]
    vel_baseline = state_dsgp4[:, 1]

    # --- Custom SGP4 ---
    tensor_single = tle_decode(tle_str)
    pos_custom, vel_custom = compiled_propagate(
        tsinces,
        bstar=tensor_single[0],
        ndot=tensor_single[1],
        nddot=tensor_single[2],
        ecco=tensor_single[3],
        argpo=tensor_single[4],
        inclo=tensor_single[5],
        mo=tensor_single[6],
        no_kozai=tensor_single[7],
        nodeo=tensor_single[8],
    )

    # --- Assertions ---
    pos_diff = torch.max(torch.abs(pos_baseline - pos_custom)).item()
    vel_diff = torch.max(torch.abs(vel_baseline - vel_custom)).item()

    assert pos_diff < 1e-9, f"Position difference {pos_diff} exceeds 1e-9 tolerance"
    assert vel_diff < 1e-9, f"Velocity difference {vel_diff} exceeds 1e-9 tolerance"


def test_batch_tle_propagation(tsinces):
    """Test batch TLE propagation accuracy against dSGP4."""
    tles_strs = [TLE_ISS, TLE_DEBRIS]
    num_steps = len(tsinces)
    num_tles = len(tles_strs)

    # --- dSGP4 Baseline ---
    tles_obj = [dsgp4.tle.TLE(tle) for tle in tles_strs]
    tles_flat = []
    for tle in tles_obj:
        tles_flat += [tle] * num_steps

    tsinces_flat = torch.cat([tsinces] * num_tles)
    _, tle_batch = dsgp4.initialize_tle(tles_flat)
    states_dsgp4 = dsgp4.propagate_batch(tle_batch, tsinces_flat)

    pos_baseline = states_dsgp4[:, 0].view(num_tles, num_steps, 3)
    vel_baseline = states_dsgp4[:, 1].view(num_tles, num_steps, 3)

    # --- Custom SGP4 ---
    tensor_batch = batch_decode(tles_strs)
    pos_custom, vel_custom = compiled_propagate(
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

    # --- Assertions ---
    pos_diff = torch.max(torch.abs(pos_baseline - pos_custom)).item()
    vel_diff = torch.max(torch.abs(vel_baseline - vel_custom)).item()

    assert pos_diff < 1e-9, f"Batch position difference {pos_diff} exceeds 1e-9"
    assert vel_diff < 1e-9, f"Batch velocity difference {vel_diff} exceeds 1e-9"
