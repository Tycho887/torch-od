from astropy.units import ds
import torch
import dsgp4
from diffod.tle import tle_decode
from diffod.functional.sgp4 import sgp4_propagate

def experiment(tle, tsinces):
    # --- dSGP4 ---
    dtle1 = dsgp4.tle.TLE(tle)
    dsgp4.initialize_tle(dtle1)
    state_dsgp4 = dsgp4.propagate(dtle1, tsinces)
    pos_dsgp4_single = state_dsgp4[:, 0] # Extract just the position (x, y, z)
    vel_dsgp4_single = state_dsgp4[:, 1]

    # --- Custom SGP4 ---
    tensor_single = tle_decode(tle)
    pos_custom_single, vel_custom_single = sgp4_propagate(
        tsinces,
        bstar=tensor_single[0], ndot=tensor_single[1], nddot=tensor_single[2],
        ecco=tensor_single[3], argpo=tensor_single[4], inclo=tensor_single[5],
        mo=tensor_single[6], no_kozai=tensor_single[7], nodeo=tensor_single[8]
    )

    pos_diff_single = torch.max(torch.abs(pos_dsgp4_single - pos_custom_single))
    vel_diff_single = torch.max(torch.abs(vel_dsgp4_single - vel_custom_single))

    return pos_diff_single, vel_diff_single

def run_comparison():
    # Force float64. SGP4 is highly sensitive to float32 truncation errors
    torch.set_default_dtype(torch.float64)

    # ---------------------------------------------------------
    # 1. Define Test Data & Time
    # ---------------------------------------------------------
    tle1_str = [
        "1 25544U 98067A   20316.40015046  .00001878  00000-0  44436-4 0  9997",
        "2 25544  51.6465 289.4354 0001961 270.2184  89.8601 15.49504104255152"
    ]
    tle2_str = [
        "1 43013U 17071A   21303.41500000  .00000112  00000-0  11234-4 0  9993",
        "2 43013  97.1234 100.5555 0011234 150.1234 210.9876 15.12345678123456"
    ]

    num_steps = 1000
    # 24 hours (1440 minutes) of propagation
    tsinces = torch.linspace(0, 1440, num_steps) 

    # ---------------------------------------------------------
    # 2. Single TLE Comparison
    # ---------------------------------------------------------
    print("=== COMPARING SINGLE TLE ===")

    for tle in [tle1_str, tle2_str]:
        pos_diff_single, vel_diff_single = experiment(tle, tsinces)
        print(f"Max Position Difference (Single): {pos_diff_single:.6e} km")
        print(f"Max Velocity Difference (Single): {vel_diff_single:.6e} km/s")


if __name__ == "__main__":
    run_comparison()