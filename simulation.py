import torch
import numpy as np
import matplotlib.pyplot as plt
from dsgp4.tle import TLE
from system import OrbitSystem
from modules import SGP4Units

# --- 1. Helper: Generate Dummy Station Data (TEME Frame) ---
def get_station_teme(t_minutes, lat_deg=55.0, lon_deg=12.0, alt_km=0.0):
    """
    Generates a simple rotating station position in TEME (Inertial).
    (Approximation: Ignores precession/nutation, assumes simple Earth rotation)
    """
    # Earth Constants
    R_e = 6378.137 # km
    omega_e = 7.292115e-5 * 60.0 # rad/min
    
    # Station Fixed (ECEF-ish)
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    r_station = R_e + alt_km
    
    # Simple z-rotation for Earth spin
    # Theta = Initial_Lon + Rotation * t
    theta = lon + omega_e * t_minutes
    
    x = r_station * np.cos(lat) * np.cos(theta)
    y = r_station * np.cos(lat) * np.sin(theta)
    z = r_station * np.sin(lat) * torch.ones_like(t_minutes)
    
    pos = torch.stack([x, y, z], dim=1)
    
    # Velocity (Derivative of pos w.r.t time)
    # v = [-y*w, x*w, 0]
    vx = -r_station * np.cos(lat) * np.sin(theta) * omega_e
    vy =  r_station * np.cos(lat) * np.cos(theta) * omega_e
    vz = torch.zeros_like(t_minutes)
    
    vel = torch.stack([vx, vy, vz], dim=1)
    
    # Convert to km/s (since omega was rad/min, vel is km/min)
    # SGP4 outputs km/s, so we must match units.
    return pos, vel / 60.0


# --- 2. Setup the "Truth" TLE ---
revs_per_day = 15.49
inclination_deg = 51.64
raan_deg = 120.0
arg_perigee_deg = 0.0
mean_anomaly_deg = 100.0

tle_dict = {
    'satellite_catalog_number': 25544,
    'epoch_year': 20,
    'epoch_days': 100.0,
    'b_star': 0.0001,
    'mean_motion': SGP4Units.rev_day_to_rad_s(revs_per_day),
    'eccentricity': 0.0003,
    'inclination': SGP4Units.deg_to_rad(inclination_deg),
    'raan': SGP4Units.deg_to_rad(raan_deg),
    'argument_of_perigee': SGP4Units.deg_to_rad(arg_perigee_deg),
    'mean_anomaly': SGP4Units.deg_to_rad(mean_anomaly_deg),
    'mean_motion_first_derivative': 0.0,
    'mean_motion_second_derivative': 0.0,
    'classification': 'U',
    'ephemeris_type': 0,
    'international_designator': '98067A',
    'revolution_number_at_epoch': 1000,
    'element_number': 999
}

truth_tle = TLE(tle_dict)


# --- 3. Simulation Configuration ---
# Simulate 2 distinct passes:
# Pass 1: t=0 to 15 mins (Index 0)
# Pass 2: t=90 to 105 mins (Index 1) - Orbital period is ~90 mins
t_pass1 = torch.linspace(0, 15, 100) # 15 mins
t_pass2 = torch.linspace(90, 105, 100) # 15 mins
t_pass3 = torch.linspace(90+90, 105+90, 100) # 15 mins

time_tensor = torch.cat([t_pass1, t_pass2, t_pass3])
contact_indices = torch.cat([torch.zeros(100), torch.ones(100), 2*torch.ones(100)]).long()

# Generate Station Ephemerides for these times
station_pos, station_vel = get_station_teme(time_tensor, lat_deg=50.0, lon_deg=10.0)
station_data = {'pos': station_pos, 'vel': station_vel}


# --- 4. Initialize the "System" ---
# Note: We use the 'OrbitSystem' class defined in the previous response
sim_system = OrbitSystem(truth_tle, station_data, num_contacts=3)


# --- 5. Inject "Secret" Biases (Optional) ---
# To make the test realistic, let's manually drift the clock for Pass 1
# Access the embedding layer directly
with torch.no_grad():
    # Set Bias for Pass 0 to +1500 Hz
    sim_system.sensor.bias_embedding.weight[0] = 1500.0 
    # Set Bias for Pass 1 to -500 Hz
    sim_system.sensor.bias_embedding.weight[1] = -500.0
    sim_system.sensor.bias_embedding.weight[2] = 500.0


# --- 6. Run Simulation (Forward Pass) ---
sim_system.eval() # Good practice, though mostly irrelevant for this specific architecture
with torch.no_grad():
    # This calls: OrbitLayer -> SGP4Layer -> DopplerSensor
    clean_doppler = sim_system(time_tensor, contact_indices)


# --- 7. Add Measurement Noise ---
# Add Gaussian white noise (sigma = 50 Hz)
noise_sigma = 5.0
noise = torch.randn_like(clean_doppler) * noise_sigma
noisy_doppler = clean_doppler + noise


# --- 8. Visualize ---
print(f"Generated {len(noisy_doppler)} data points.")
print(f"Pass 1 Mean Doppler: {clean_doppler[:100].mean():.2f} Hz (Includes +1500 bias)")

plt.figure(figsize=(10, 5))
plt.plot(time_tensor.numpy(), noisy_doppler.numpy(), '.', label='Simulated Data (Noisy)', alpha=0.5)
plt.plot(time_tensor.numpy(), clean_doppler.numpy(), 'k-', label='Truth (Clean)', linewidth=1)
plt.xlabel('Time (minutes from Epoch)')
plt.ylabel('Doppler Shift (Hz)')
plt.title('Simulated Doppler Data with Per-Pass Biases')
plt.legend()
plt.grid(True)
plt.savefig('simulated_doppler.png', dpi=300)
plt.close()