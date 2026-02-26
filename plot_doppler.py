import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from diffod.functional.system import DopplerMeasurement, GPSInterpolator, MeasurementPipeline
from diffod.gse import station_teme_preprocessor
from diffod.utils import load_gmat_csv_block
from diffod.state import CalibrationSSV

def main():
    # ---------------------------------------------------------
    # 1. Setup & Load Data
    # ---------------------------------------------------------
    print("Loading Data...")
    epoch = 1762165047
    center_freq = 1.707e9
    target_device = torch.device("cpu")
    
    # Example station dictionary
    stations = {0: np.array([15.376932, 78.228874, 0.463])}

    # Load Doppler Telemetry
    period_telemetry = pl.read_parquet("data/period_telemetry.parquet")
    times_unix = torch.tensor(period_telemetry["timestamp"].to_numpy(), dtype=torch.float64, device=target_device)
    doppler_obs = torch.tensor(period_telemetry["Doppler_Hz"].to_numpy(), dtype=torch.float64, device=target_device)
    contacts = torch.tensor(period_telemetry["contact_index"].to_numpy(), dtype=torch.int32, device=target_device)
    st_indices = torch.zeros(len(times_unix), dtype=torch.int32, device=target_device)

    # Load Ground Truth GPS
    t_gps_raw, r_gps_raw, v_gps_raw = load_gmat_csv_block(
        file_path="data/AWS_high_frequency.csv",
        tle_epoch_unix=float(torch.mean(times_unix)),
        block_sec=86400 * 1,
    )

    # ---------------------------------------------------------
    # 2. Filter & Align Timestamps
    # ---------------------------------------------------------
    gps_start, gps_end = t_gps_raw.min(), t_gps_raw.max()
    valid_mask = (times_unix >= gps_start) & (times_unix <= gps_end)

    times_unix = times_unix[valid_mask]
    doppler_obs = doppler_obs[valid_mask]
    contacts = contacts[valid_mask]
    st_indices = st_indices[valid_mask]

    # Center times for numerical stability in interpolation
    T_mean = t_gps_raw.mean()
    t_gps_centered = (t_gps_raw - T_mean).to(target_device)
    t_obs_centered = (times_unix - T_mean).to(target_device)

    # ---------------------------------------------------------
    # 3. Precompute Ground Station Ephemeris
    # ---------------------------------------------------------
    print("Preprocessing Ground Station Ephemeris...")
    station_pos_cpu, station_vel_cpu = station_teme_preprocessor(
        times_s=times_unix.numpy(), 
        station_ids=st_indices.numpy(),
        id_to_station=stations,
        dtype=torch.float64,
        device=target_device,
    )
    st_pos = station_pos_cpu.to(target_device)
    st_vel = station_vel_cpu.to(target_device)

    # ---------------------------------------------------------
    # 4. Construct Forward Pipeline (No Solvers/Biases)
    # ---------------------------------------------------------
    # Use the SSV purely for architectural compatibility, but don't add bias groups
    ssv = CalibrationSSV(num_measurements=len(times_unix), fit_time_offset=True)
    
    # Create a zero-state tensor (0 time offset, 0 biases)
    x_zero = torch.zeros(ssv.current_dim, dtype=torch.float64, device=target_device)

    interpolator = GPSInterpolator(
        ssv=ssv, 
        t_gps_ref=t_gps_centered, 
        r_gps_ref=r_gps_raw.to(target_device), 
        v_gps_ref=v_gps_raw.to(target_device)
    )

    doppler_model = DopplerMeasurement(ssv=ssv, bias_group=None)
    model = MeasurementPipeline(propagator=interpolator, measurement_model=doppler_model)

    # ---------------------------------------------------------
    # 5. Execute Forward Pass (Expected Doppler)
    # ---------------------------------------------------------
    print("Computing Expected Doppler from Interpolated GPS...")
    with torch.no_grad():
        expected_doppler = model(
            x=x_zero, 
            st_pos=st_pos, 
            st_vel=st_vel, 
            tsince=t_obs_centered,
            center_freq=center_freq
        )

    # ---------------------------------------------------------
    # 6. Visualization
    # ---------------------------------------------------------
    plot_expected_vs_measured(
        t_obs=(times_unix - T_mean),# / 60.0, # Plot in minutes since epoch
        measured=doppler_obs,
        expected=expected_doppler,
        contacts=contacts
    )

def plot_expected_vs_measured(t_obs, measured, expected, contacts):
    """Clean plotting utility for forward-pass evaluation."""
    t_np = t_obs.detach().cpu().numpy()
    meas_np = measured.detach().cpu().numpy()
    exp_np = expected.detach().cpu().numpy()
    contacts_np = contacts.detach().cpu().numpy()
    
    unique_contacts = np.unique(contacts_np)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    for c in unique_contacts:
        mask = contacts_np == c
        sort_idx = np.argsort(t_np[mask])
        
        t_c = t_np[mask][sort_idx]
        meas_c = meas_np[mask][sort_idx]
        exp_c = exp_np[mask][sort_idx]
        res_c = meas_c - exp_c
        
        label_meas = 'Measured Data (Raw)' if c == unique_contacts[0] else ""
        label_exp = 'Expected (Interpolated GPS)' if c == unique_contacts[0] else ""
        
        axes[0].plot(t_c, meas_c, 'k.', label=label_meas, alpha=0.4, markersize=4)
        axes[0].plot(t_c, exp_c, 'r-', label=label_exp, linewidth=2, alpha=0.8)
        axes[1].plot(t_c, res_c, 'b.', alpha=0.5, markersize=4)
        
    axes[0].set_ylabel("Doppler Shift (Hz)")
    axes[0].set_title("Expected vs. Measured Doppler")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.7)
    
    axes[1].set_ylabel("Residual (Hz)")
    axes[1].set_xlabel("Time Since Epoch (Minutes)")
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    axes[1].grid(True, linestyle=':', alpha=0.7)
    
    rms_error = np.sqrt(np.mean((meas_np - exp_np)**2))
    axes[1].set_title(f"Uncalibrated Residuals (Overall RMS: {rms_error:.2f} Hz)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()