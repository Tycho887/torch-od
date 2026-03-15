import torch
import dsgp4
from diffod.tle import tle_decode, batch_decode
from diffod.functional.sgp4 import sgp4_propagate

compiled_propagate = torch.jit.script(sgp4_propagate)

def experiment(tle, tsinces):
    # --- dSGP4 ---
    dtle1 = dsgp4.tle.TLE(tle)
    dsgp4.initialize_tle(dtle1)
    state_dsgp4 = dsgp4.propagate(dtle1, tsinces)
    pos_dsgp4_single = state_dsgp4[:, 0] # Extract just the position (x, y, z)
    vel_dsgp4_single = state_dsgp4[:, 1]

    # --- Custom SGP4 ---
    tensor_single = tle_decode(tle)

    pos_custom_single, vel_custom_single = compiled_propagate(
        tsinces,
        bstar=tensor_single[0], ndot=tensor_single[1], nddot=tensor_single[2],
        ecco=tensor_single[3], argpo=tensor_single[4], inclo=tensor_single[5],
        mo=tensor_single[6], no_kozai=tensor_single[7], nodeo=tensor_single[8]
    )

    pos_diff_single = torch.max(torch.abs(pos_dsgp4_single - pos_custom_single))
    vel_diff_single = torch.max(torch.abs(vel_dsgp4_single - vel_custom_single))

    return pos_diff_single, vel_diff_single

def batch_experiment(tles_strs, tsinces):
    num_steps = len(tsinces)
    num_tles = len(tles_strs)

    # --- dSGP4 Batch ---
    tles_obj = [dsgp4.tle.TLE(tle) for tle in tles_strs]
    tles_flat = []
    
    # dSGP4 requires duplicating the TLE for every timestep
    for tle in tles_obj:
        tles_flat += [tle] * num_steps
        
    tsinces_flat = torch.cat([tsinces] * num_tles)
    
    # Initialize and propagate
    _, tle_batch = dsgp4.initialize_tle(tles_flat)
    states_dsgp4 = dsgp4.propagate_batch(tle_batch, tsinces_flat)
    
    # Reshape dSGP4 flat output to (N_sats, N_steps, 3)
    pos_dsgp4_batch = states_dsgp4[:, 0].view(num_tles, num_steps, 3)
    vel_dsgp4_batch = states_dsgp4[:, 1].view(num_tles, num_steps, 3)

    # --- Custom SGP4 Batch ---
    tensor_batch = batch_decode(tles_strs) # Shape: (N, 9)

    # Unsqueeze parameters to (N, 1) so PyTorch broadcasts against tsinces (T,)
    pos_custom_batch, vel_custom_batch = compiled_propagate(
        tsinces,
        bstar=tensor_batch[:, 0].unsqueeze(1),
        ndot=tensor_batch[:, 1].unsqueeze(1),
        nddot=tensor_batch[:, 2].unsqueeze(1),
        ecco=tensor_batch[:, 3].unsqueeze(1),
        argpo=tensor_batch[:, 4].unsqueeze(1),
        inclo=tensor_batch[:, 5].unsqueeze(1),
        mo=tensor_batch[:, 6].unsqueeze(1),
        no_kozai=tensor_batch[:, 7].unsqueeze(1),
        nodeo=tensor_batch[:, 8].unsqueeze(1)
    )

    pos_diff_batch = torch.max(torch.abs(pos_dsgp4_batch - pos_custom_batch))
    vel_diff_batch = torch.max(torch.abs(vel_dsgp4_batch - vel_custom_batch))

    return pos_diff_batch, vel_diff_batch


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
    for i, tle in enumerate([tle1_str, tle2_str]):
        pos_diff_single, vel_diff_single = experiment(tle, tsinces)
        print(f"TLE {i+1} - Max Position Difference: {pos_diff_single:.6e} km")
        print(f"TLE {i+1} - Max Velocity Difference: {vel_diff_single:.6e} km/s")

    # ---------------------------------------------------------
    # 3. Batch TLE Comparison
    # ---------------------------------------------------------
    print("\n=== COMPARING BATCH TLE ===")
    pos_diff_batch, vel_diff_batch = batch_experiment([tle1_str, tle2_str], tsinces)
    
    print(f"Batch - Max Position Difference: {pos_diff_batch:.6e} km")
    print(f"Batch - Max Velocity Difference: {vel_diff_batch:.6e} km/s")


if __name__ == "__main__":
    run_comparison()