import torch
import h5py
import math
from pathlib import Path

from torch.nn.qat import dynamic
from sgp4 import sgp4_propagate

# Import your custom differentiable SGP4
# from sgp4 import sgp4_propagate 

def generate_sso_dataset(num_trajectories=500, steps_per_trajectory=200, save_path="sso_dataset.h5"):
    # 1. Define the simulation timeframe
    # SGP4 typically operates in minutes since epoch.
    # 200 minutes covers roughly 2 full orbits in LEO.
    timestamps = torch.linspace(0, 200, steps_per_trajectory)
    
    # 2. Define SSO Parameter Bounds
    # Mean Motion (no_kozai): ~14.5 to 15.5 revs/day (approx 500 km to 800 km altitude)
    # Inclination (inclo): ~97.0 to 98.5 degrees for SSO
    # Eccentricity (ecco): Near circular
    # BSTAR: Varies based on drag/satellite area-to-mass
    
    # Conversion factors for SGP4 standard inputs
    rev_per_day_to_rad_per_min = (2 * math.pi) / 1440.0
    deg_to_rad = math.pi / 180.0

    # Sample parameters uniformly within SSO bounds
    # Using torch.rand to generate shapes of (num_trajectories, 1) for broadcasting
    no_kozai_revs = 14.5 + torch.rand(num_trajectories, 1) * 1.0
    no_kozai = no_kozai_revs * rev_per_day_to_rad_per_min
    
    inclo_deg = 97.0 + torch.rand(num_trajectories, 1) * 1.5
    inclo = inclo_deg * deg_to_rad
    
    ecco = 0.0001 + torch.rand(num_trajectories, 1) * 0.004
    bstar = 1e-6 + torch.rand(num_trajectories, 1) * (1e-4 - 1e-6)
    
    # Randomize ascending node, argument of perigee, and mean anomaly across the full 2*pi range
    nodeo = torch.rand(num_trajectories, 1) * 2 * math.pi
    argpo = torch.rand(num_trajectories, 1) * 2 * math.pi
    mo = torch.rand(num_trajectories, 1) * 2 * math.pi
    
    # Perturbation terms (typically negligible or 0 for initialization, but included for completeness)
    ndot = torch.zeros(num_trajectories, 1)
    nddot = torch.zeros(num_trajectories, 1)
    
    # 3. Propagate Trajectories
    # Assuming your compiled_propagator broadcasts the (N, 1) parameters against the (T,) timestamps
    # to return tensors of shape (num_trajectories, steps_per_trajectory, 3)
    print(f"Propagating {num_trajectories} SSO trajectories...")
    
    from time import time

    # compiled_propagator = torch.compile(sgp4_propagate, dynamic=True)
    t0 = time()
    pos, vel = sgp4_propagate(
        timestamps, 
        bstar=bstar, 
        ndot=ndot, 
        nddot=nddot, 
        ecco=ecco, 
        argpo=argpo, 
        inclo=inclo, 
        mo=mo, 
        no_kozai=no_kozai, 
        nodeo=nodeo
    )
    t1 = time()
    print(f"Time taken: {1e3*(t1 - t0):.3f} ms")

    print(torch.linalg.norm(pos, dim=-1), torch.linalg.norm(vel, dim=-1))
    
    # 4. Save to HDF5
    print(f"Saving dataset to {save_path}...")
    with h5py.File(save_path, 'w') as f:
        # Save time information
        f.create_dataset('timestamps', data=timestamps.numpy())
        
        # Save state data
        f.create_dataset('positions', data=pos.detach().numpy())
        f.create_dataset('velocities', data=vel.detach().numpy())
        
        # Save the underlying parameters for analysis/validation
        params_group = f.create_group('parameters')
        params_group.create_dataset('no_kozai_rad_min', data=no_kozai.numpy())
        params_group.create_dataset('inclo_rad', data=inclo.numpy())
        params_group.create_dataset('ecco', data=ecco.numpy())
        params_group.create_dataset('bstar', data=bstar.numpy())
        params_group.create_dataset('nodeo', data=nodeo.numpy())
        params_group.create_dataset('argpo', data=argpo.numpy())
        params_group.create_dataset('mo', data=mo.numpy())

    print("Dataset generation complete.")

if __name__ == "__main__":
    generate_sso_dataset()