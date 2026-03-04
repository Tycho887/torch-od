from astropy import conf
import torch
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
import dsgp4
from astropy.time import Time
import diffod.state as state
from diffod.utils import unix_to_mjd, compute_observability_metrics

def compute_ric_residuals(
    x_state: torch.Tensor,
    propagator: torch.nn.Module,
    t_gps: torch.Tensor, 
    r_gps: torch.Tensor, 
    v_gps: torch.Tensor, 
    tle_epoch_unix: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # 1. Convert GPS inputs from meters back to kilometers for SGP4 math
    r_gps_km = r_gps.to(torch.float64) 
    v_gps_km = v_gps.to(torch.float64) 
    t_gps = t_gps.to(torch.float64)

    # 2. Propagate state using your PyTorch functional/ML block
    t_since_mins = (t_gps - tle_epoch_unix) / 60.0
    
    # Bypass dsgp4; use the exact PyTorch graph evaluated by the solver
    with torch.no_grad():
        r_calc_km, v_calc_km = propagator(x=x_state, tsince=t_since_mins)

    # 3. Calculate Cartesian Errors (Calculated - Truth)
    delta_r = r_calc_km - r_gps_km
    delta_v = v_calc_km - v_gps_km

    # 4. Construct RIC Basis Vectors (using GPS state as the reference)
    R_hat = r_gps_km / torch.norm(r_gps_km, dim=1, keepdim=True)
    W_vec = torch.cross(r_gps_km, v_gps_km, dim=1)
    W_hat = W_vec / torch.norm(W_vec, dim=1, keepdim=True)
    S_hat = torch.cross(W_hat, R_hat, dim=1)

    # 5. Project Cartesian errors into the RIC frame
    pos_res_R = (delta_r * R_hat).sum(dim=1).unsqueeze(1)
    pos_res_A = (delta_r * S_hat).sum(dim=1).unsqueeze(1)
    pos_res_C = (delta_r * W_hat).sum(dim=1).unsqueeze(1)
    
    vel_res_R = (delta_v * R_hat).sum(dim=1).unsqueeze(1)
    vel_res_A = (delta_v * S_hat).sum(dim=1).unsqueeze(1)
    vel_res_C = (delta_v * W_hat).sum(dim=1).unsqueeze(1)

    pos_ric = torch.cat([pos_res_R, pos_res_A, pos_res_C], dim=1)
    vel_ric = torch.cat([vel_res_R, vel_res_A, vel_res_C], dim=1)

    return t_since_mins, pos_ric, vel_ric

import numpy as np
import matplotlib.pyplot as plt
import torch

import matplotlib.pyplot as plt
import torch

def plot_single_orbit_ric_residuals(
    t_mins: torch.Tensor, 
    pos_ric: torch.Tensor, 
    vel_ric: torch.Tensor,
    title: str = "6-Pass 2-DOF Estimator: RIC Residuals Post-Fit"
):
    """
    Plots the raw RIC position and velocity residuals for a single computed orbit.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    components = ['Radial', 'In-track', 'Cross-track']
    
    # Convert tensors to numpy arrays and minutes to hours
    t_hours = t_mins.detach().cpu().numpy() / 60.0
    pos_np = pos_ric.detach().cpu().numpy()
    vel_np = vel_ric.detach().cpu().numpy()
    
    # Use distinct colors for each axis to make it visually appealing
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    
    for i in range(3):
        # Position Row
        axes[0, i].plot(t_hours, pos_np[:, i], color=colors[i], linewidth=2)
        axes[0, i].set_title(f'{components[i]} Error')
        axes[0, i].grid(True, linestyle='--', alpha=0.6)
        axes[0, i].axhline(0, color='black', linewidth=1, linestyle='-') # Add a zero-line
        
        # Velocity Row
        axes[1, i].plot(t_hours, vel_np[:, i], color=colors[i], linewidth=2)
        axes[1, i].grid(True, linestyle='--', alpha=0.6)
        axes[1, i].axhline(0, color='black', linewidth=1, linestyle='-')
        axes[1, i].set_xlabel('Time Since Fit Epoch (Hours)')
        
    # Set y-labels only on the first column to save space
    axes[0, 0].set_ylabel('Position Error (km)')
    axes[1, 0].set_ylabel('Velocity Error (km/s)')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def print_ric_residual_summary(results_dict: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    """
    Prints a formatted summary of RMS errors in the RIC frame for each solver result.
    """
    components = ['Radial', 'Along-track', 'Cross-track']
    header = f"{'Solver Name':<25} | {'Axis':<12} | {'Pos RMS (km)':<14} | {'Vel RMS (km/s)':<14}"
    divider = "-" * len(header)

    print("\n" + divider)
    print(header)
    print(divider)

    for name, (pos_ric, vel_ric) in results_dict.items():
        # Calculate RMS: sqrt(mean(x^2))
        pos_rms = torch.sqrt(torch.mean(pos_ric**2, dim=0))
        vel_rms = torch.sqrt(torch.mean(vel_ric**2, dim=0))

        for i, comp in enumerate(components):
            # Only print the name on the first row of each solver block for readability
            row_name = name if i == 0 else ""
            print(f"{row_name:<25} | {comp:<12} | {pos_rms[i]:<14.6f} | {vel_rms[i]:<14.6f}")
        print(divider)

# Usage in your script:
def plot_ric_residuals(
    t_mins: torch.Tensor, 
    results_dict: dict[str, tuple[torch.Tensor, torch.Tensor]],
):
    t_plot = t_mins.detach().cpu().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    components = ['Radial', 'Along-track', 'Cross-track']
    
    styles = {
        # "Initial TLE": {"color": "red", "linestyle": "--", "alpha": 0.7, "linewidth": 2},
        "Gauss-Newton (WGN)": {"color": "blue", "linestyle": "-", "alpha": 0.8, "linewidth": 2},
        "Exact Newton": {"color": "purple", "linestyle": ":", "alpha": 0.8, "linewidth": 2.5},
        "Consider Covariance (CCA)": {"color": "green", "linestyle": "-.", "alpha": 1.0, "linewidth": 2},
    }

    for name, (pos_ric, vel_ric) in results_dict.items():
        pos_np = pos_ric.detach().cpu().numpy()
        vel_np = vel_ric.detach().cpu().numpy()
        style = styles.get(name, {"linestyle": "-", "alpha": 0.8})

        for i in range(3):
            axes[0, i].plot(t_plot, pos_np[:, i], label=name, **style)
            axes[1, i].plot(t_plot, vel_np[:, i], label=name, **style)

    for i, comp in enumerate(components):
        axes[0, i].set_title(f'{comp} Error')
        axes[0, i].set_ylabel('Position Error (km)' if i == 0 else '')
        axes[1, i].set_ylabel('Velocity Error (km/s)' if i == 0 else '')
        axes[1, i].set_xlabel('Time Since Epoch (Minutes)')
        
        for row in range(2):
            axes[row, i].grid(True, linestyle=':', alpha=0.7)
            if i == 0 and row == 0:
                axes[row, i].legend(loc='best')

    plt.suptitle('GPS vs TLE Residuals in RIC Frame', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_calibrated_doppler(
    t_obs: torch.Tensor, 
    doppler_obs: torch.Tensor, 
    doppler_pred: torch.Tensor, 
    contacts: torch.Tensor
):
    """
    Plots the measured vs. calibrated Doppler curves and their residuals, 
    segmented by contact passes.
    """
    # 1. Move tensors to CPU and convert to NumPy
    t_np = t_obs.detach().cpu().numpy()
    obs_np = doppler_obs.detach().cpu().numpy()
    pred_np = doppler_pred.detach().cpu().numpy()
    contacts_np = contacts.detach().cpu().numpy()
    
    unique_contacts = np.unique(contacts_np)
    
    # 2. Setup the plot canvas
    fig, axes = plt.subplots(
        2, 1, 
        figsize=(12, 8), 
        sharex=True, 
        gridspec_kw={'height_ratios': [2, 1]}
    )
    
    # 3. Iterate through each pass to prevent connecting lines across time gaps
    for c in unique_contacts:
        mask = contacts_np == c
        
        # Sort chronologically within the pass
        sort_idx = np.argsort(t_np[mask])
        t_c = t_np[mask][sort_idx]
        obs_c = obs_np[mask][sort_idx]
        pred_c = pred_np[mask][sort_idx]
        res_c = obs_c - pred_c
        
        # Plot 1: Absolute Curves
        # Only add the label to the first pass to avoid legend duplication
        label_obs = 'Measured Data' if c == unique_contacts[0] else ""
        label_pred = 'Calibrated Model' if c == unique_contacts[0] else ""
        
        axes[0].plot(t_c, obs_c, 'k.', label=label_obs, alpha=0.4, markersize=4)
        axes[0].plot(t_c, pred_c, 'r-', label=label_pred, linewidth=2, alpha=0.8)
        
        # Plot 2: Residuals
        axes[1].plot(t_c, res_c, 'b.', alpha=0.5, markersize=4)
        
    # 4. Formatting
    axes[0].set_ylabel("Doppler Shift (Hz)")
    axes[0].set_title("Measured vs. Calibrated Doppler Curves")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.7)
    
    axes[1].set_ylabel("Residual (Hz)")
    axes[1].set_xlabel("Time Since Epoch (Minutes)")
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    axes[1].grid(True, linestyle=':', alpha=0.7)
    
    # Calculate overall RMS for the title
    rms_error = np.sqrt(np.mean((obs_np - pred_np)**2))
    axes[1].set_title(f"Residuals (Overall RMS: {rms_error:.2f} Hz)")
    
    plt.tight_layout()
    plt.show()

# import torch
# import numpy as np
# import matplotlib.pyplot as plt

def plot_residual_diagnostics(residuals: torch.Tensor, lags: int = 50, num_bins: int = 30):
    """
    Generates an Autocorrelation Function (ACF) plot and a distribution plot 
    with a Chi-Squared goodness-of-fit test for normality.
    """
    # Detach and convert to 1D numpy array
    res = residuals.detach().cpu().numpy().flatten()
    N = len(res)
    
    # 1. Setup Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ---------------------------------------------------------
    # Plot 1: Autocorrelation Function (ACF)
    # ---------------------------------------------------------
    ax_acf = axes[0]
    
    # Calculate and plot ACF (normed=True scales lag-0 to 1.0)
    ax_acf.acorr(res, maxlags=lags, usevlines=True, normed=True, color='blue', lw=2)
    
    # 95% Confidence Intervals for White Noise: +/- 1.96 / sqrt(N)
    conf_interval = 1.96 / np.sqrt(N)
    ax_acf.axhline(conf_interval, color='red', linestyle='--', alpha=0.7)
    ax_acf.axhline(-conf_interval, color='red', linestyle='--', alpha=0.7)
    ax_acf.axhline(0, color='black', lw=1)
    
    ax_acf.set_title("Autocorrelation Function (ACF)")
    ax_acf.set_xlabel("Lag")
    ax_acf.set_ylabel("Autocorrelation")
    ax_acf.grid(True, linestyle=':', alpha=0.7)
    
    # The acorr function plots negative lags too; limit to positive for standard view
    ax_acf.set_xlim([0, lags]) 
    
    # ---------------------------------------------------------
    # Plot 2: Distribution & Chi-Squared Test
    # ---------------------------------------------------------
    ax_dist = axes[1]
    
    # Calculate sample statistics
    mu, std = np.mean(res), np.std(res)
    
    # Create bins and calculate observed frequencies
    observed_freq, bin_edges = np.histogram(res, bins=num_bins)
    
    # Calculate expected frequencies using the Normal CDF over the bins
    cdf_values = stats.norm.cdf(bin_edges, loc=mu, scale=std)
    expected_prob = np.diff(cdf_values)
    expected_freq = expected_prob * N
    
    # Normalize expected frequencies to exactly match the sum of observed 
    # (Required by scipy.stats.chisquare to prevent floating point assertion errors)
    expected_freq = expected_freq * (observed_freq.sum() / expected_freq.sum())
    
    # Chi-Squared Test
    # ddof=2 because we estimated 2 parameters (mu, std) from the sample data
    chi2_stat, p_val = stats.chisquare(f_obs=observed_freq, f_exp=expected_freq, ddof=2)
    
    # Plot Observed Histogram
    ax_dist.hist(res, bins=bin_edges, density=True, alpha=0.6, color='gray', 
                 edgecolor='black', label='Observed Residuals')
    
    # Plot Theoretical Normal PDF
    x_pdf = np.linspace(bin_edges[0], bin_edges[-1], 200)
    y_pdf = stats.norm.pdf(x_pdf, mu, std)
    ax_dist.plot(x_pdf, y_pdf, 'r-', lw=2, label=f'Normal PDF ($\mu$={mu:.2f}, $\sigma$={std:.2f})')
    
    # Annotate with Chi-Squared Results
    stat_text = f"$\chi^2$ Stat: {chi2_stat:.2f}\np-value: {p_val:.4e}"
    ax_dist.text(0.95, 0.95, stat_text, transform=ax_dist.transAxes, 
                 fontsize=11, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_dist.set_title("Residual Distribution & $\chi^2$ Test")
    ax_dist.set_xlabel("Residual Magnitude (Hz)")
    ax_dist.set_ylabel("Density")
    ax_dist.legend(loc='upper left')
    ax_dist.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    return chi2_stat, p_val

# import torch
# import numpy as np
# import matplotlib.pyplot as plt

def plot_pass_biases(freq_biases_hz: torch.Tensor | np.ndarray | list, 
                     time_biases_sec: torch.Tensor | np.ndarray | list, 
                     pass_indices: list | np.ndarray = None):
    """
    Plots the estimated frequency and timing biases for each contact pass.
    Useful for system calibration diagnostics prior to Orbit Determination.
    """
    # Detach and convert to numpy arrays if they are PyTorch tensors
    if isinstance(freq_biases_hz, torch.Tensor):
        freq_biases_hz = freq_biases_hz.detach().cpu().numpy()
    if isinstance(time_biases_sec, torch.Tensor):
        time_biases_sec = time_biases_sec.detach().cpu().numpy()
        
    # Ensure they are flat arrays
    freq_biases_hz = np.atleast_1d(np.squeeze(freq_biases_hz))
    time_biases_sec = np.atleast_1d(np.squeeze(time_biases_sec))
        
    N_passes = len(freq_biases_hz)
    if pass_indices is None:
        pass_indices = np.arange(N_passes)
        
    # Setup Figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # ---------------------------------------------------------
    # 0: Compute mean and variance
    # ---------------------------------------------------------
    freq_mean = np.mean(freq_biases_hz)
    freq_std = np.std(freq_biases_hz)

    time_mean = np.mean(time_biases_sec)
    time_std = np.std(time_biases_sec)
    
    # ---------------------------------------------------------
    # Plot 1: Frequency Biases
    # ---------------------------------------------------------
    axes[0].plot(pass_indices, freq_biases_hz, marker='o', linestyle='-', 
                 color='#1f77b4', linewidth=2, markersize=6)
    axes[0].set_ylabel("Frequency Bias (Hz)")
    axes[0].set_title("Calibrated Per-Pass Biases")
    axes[0].grid(True, linestyle=':', alpha=0.7)
    axes[0].hlines(freq_mean, xmin=pass_indices[0], xmax=pass_indices[-1], label=f"Mean: {freq_mean:.3f}, std: {freq_std:.3f}")
    axes[0].legend()

    # ---------------------------------------------------------
    # Plot 2: Time Biases
    # ---------------------------------------------------------
    axes[1].plot(pass_indices, time_biases_sec, marker='s', linestyle='-', 
                 color='#d62728', linewidth=2, markersize=6)
    axes[1].set_ylabel("Time Bias (Seconds)")
    axes[1].set_xlabel("Contact Pass Index")
    axes[1].grid(True, linestyle=':', alpha=0.7)
    axes[1].hlines(time_mean, xmin=pass_indices[0], xmax=pass_indices[-1], label=f"Mean: {time_mean:.3f}, std: {time_std:.3f}")
    axes[1].legend()
    # Force the x-axis to show integer ticks for discrete passes
    axes[1].set_xticks(pass_indices)
    
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import torch
import diffod.functional.system as system

def compare_doppler_models(
    t_dopp: torch.Tensor, 
    d_dopp: torch.Tensor, 
    t_gps: torch.Tensor, 
    r_gps: torch.Tensor, 
    v_gps: torch.Tensor,
    x_tle_init: torch.Tensor, 
    x_gps_fit: torch.Tensor, 
    ssv, 
    station_model, 
    T_mean: float, 
    center_freq: float
):
    """
    Evaluates and plots the residuals of the measured Doppler against 
    Initial TLE, GPS-Fitted TLE, and Direct GPS Interpolation.
    """
    t_since_dopp = t_dopp - T_mean
    
    # 1. Setup SGP4 Pipeline
    prop_sgp4 = system.SGP4(ssv=ssv, use_pretrained_model=False)
    meas_dopp = system.DopplerMeasurement(ssv=ssv, station_model=station_model)
    pipe_sgp4 = system.MeasurementPipeline(propagator=prop_sgp4, measurement_model=meas_dopp)
    
    # 2. Setup GPS Interpolator Pipeline
    interpolator = system.GPSInterpolator(
        ssv=ssv,
        t_gps_ref=(t_gps - T_mean),
        r_gps_ref=r_gps,
        v_gps_ref=v_gps
    )
    pipe_interp = system.MeasurementPipeline(propagator=interpolator, measurement_model=meas_dopp)
    
    # 3. Generate Predictions
    with torch.no_grad():
        d_pred_init = pipe_sgp4(x=x_tle_init, tsince=t_since_dopp, epoch=T_mean, center_freq=center_freq)
        d_pred_fit = pipe_sgp4(x=x_gps_fit, tsince=t_since_dopp, epoch=T_mean, center_freq=center_freq)
        d_pred_interp = pipe_interp(x=x_gps_fit, tsince=t_since_dopp, epoch=T_mean, center_freq=center_freq)
        
    # 4. Calculate Residuals (Measured - Predicted)
    res_init = (d_dopp - d_pred_init).cpu().numpy()
    res_fit = (d_dopp - d_pred_fit).cpu().numpy()
    res_interp = (d_dopp - d_pred_interp).cpu().numpy()
    t_plot = (t_since_dopp / 60.0).cpu().numpy() # Plot in minutes from epoch

    # 5. Plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top Plot: Absolute Doppler
    axs[0].scatter(t_plot, d_dopp.cpu().numpy(), color='black', label='Measured (GMAT)', s=10, alpha=0.6)
    axs[0].plot(t_plot, d_pred_init.cpu().numpy(), color='red', label='Initial TLE', linewidth=1)
    axs[0].plot(t_plot, d_pred_fit.cpu().numpy(), color='blue', label='GPS-Fit TLE', linewidth=1)
    axs[0].plot(t_plot, d_pred_interp.cpu().numpy(), color='green', label='GPS Interpolated', linewidth=1, linestyle='--')
    axs[0].set_ylabel("Doppler Shift (Hz)")
    axs[0].set_title("Absolute Doppler Frequencies")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Bottom Plot: Residuals
    axs[1].scatter(t_plot, res_init, color='red', label=f'Init TLE RMS: {np.sqrt(np.mean(res_init**2)):.2f} Hz', s=5)
    axs[1].scatter(t_plot, res_fit, color='blue', label=f'GPS-Fit RMS: {np.sqrt(np.mean(res_fit**2)):.2f} Hz', s=5)
    axs[1].scatter(t_plot, res_interp, color='green', label=f'GPS Interp RMS: {np.sqrt(np.mean(res_interp**2)):.2f} Hz', s=5)
    axs[1].axhline(0, color='black', linewidth=1)
    axs[1].set_xlabel(f"Time since Epoch {T_mean} (Minutes)")
    axs[1].set_ylabel("Residuals (Hz)")
    axs[1].set_title("Doppler Residuals (Measured - Predicted)")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_uncertainty_bounds(t_gps, ensemble_ric, std_analytical):
    """
    Plots the Monte Carlo empirical spread against the Jacobian-propagated analytical bounds.
    """
    print("\n--- Plotting 3D Confidence Bounds ---")
    t_plot_mins = (t_gps - t_gps[0]).cpu().numpy() / 60.0
    
    # Calculate Empirical MC stats
    mean_mc = torch.mean(ensemble_ric, dim=0).cpu().numpy()
    std_mc = torch.std(ensemble_ric, dim=0).cpu().numpy()
    
    std_ana = std_analytical.cpu().numpy()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    labels = ['Radial', 'In-track', 'Cross-track']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i in range(3):
        ax = axes[i]
        # 1. Plot the MC Mean (Residual bias)
        ax.plot(t_plot_mins, mean_mc[:, i], color=colors[i], label='MC Mean Residual', linewidth=1.5)
        
        # 2. Plot the MC Empirical Spread (+/- 3 Sigma Blob)
        ax.fill_between(t_plot_mins, 
                        mean_mc[:, i] - 3*std_mc[:, i], 
                        mean_mc[:, i] + 3*std_mc[:, i], 
                        color=colors[i], alpha=0.25, label=r'MC $\pm 3\sigma$ (Empirical)')
        
        # 3. Overlay the Analytical Bounds (Centered around the mean for direct spread comparison)
        ax.plot(t_plot_mins, mean_mc[:, i] + 3*std_ana[:, i], color='black', linestyle='--', linewidth=1.5, label=r'Analytical $\pm 3\sigma$ (Jacobian)')
        ax.plot(t_plot_mins, mean_mc[:, i] - 3*std_ana[:, i], color='black', linestyle='--', linewidth=1.5)
        
        ax.set_ylabel(f'{labels[i]} Error (km)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Time since start (minutes)')
    plt.suptitle('Orbit Determination Uncertainty: Empirical Monte Carlo vs. Analytical Jacobian', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_cross_validation(results: list[dict]):
    """Generates cross-validated bar plots for OD performance metrics."""
    df = pl.DataFrame(results)
    passes = sorted(df["num_passes"].unique().to_list())
    chunks = sorted(df["chunk_id"].unique().to_list())
    
    metrics_to_plot = [
        ("1h_rmse", "1-Hour Forecast RMSE (km)"),
        ("24h_rmse", "24-Hour Forecast RMSE (km)"),
        ("cov_frob", "Covariance Frobenius Norm")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    bar_width = 0.8 / len(chunks)
    
    for ax, (metric_key, title) in zip(axes, metrics_to_plot):
        for i, chunk_id in enumerate(chunks):
            chunk_data = df.filter(pl.col("chunk_id") == chunk_id).sort("num_passes")
            x_offsets = np.arange(len(passes)) + (i * bar_width) - (0.4 - bar_width/2)
            
            y_vals = chunk_data[metric_key].to_numpy()
            ax.bar(x_offsets, y_vals, width=bar_width, label=f"Segment {chunk_id}", alpha=0.8)
            
        ax.set_title(title)
        ax.set_xlabel("Number of Passes")
        ax.set_xticks(np.arange(len(passes)))
        ax.set_xticklabels(passes)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Add a logarithmic scale if plotting the Frobenius norm
        if "Covariance" in title:
            ax.set_yscale('log')
            
    axes[-1].legend(title="Data Segments")
    plt.tight_layout()
    plt.show()

def plot_segment_ric_residuals(
    segment_results_dict: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """
    Plots the RIC position and velocity errors for different dataset segments.
    Expects segment_results_dict to map 'Segment Name' -> (t_mins, pos_ric, vel_ric).
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    components = ['Radial', 'Along-track', 'Cross-track']
    
    # Use a standard colormap to dynamically handle any number of segments
    colors = plt.cm.tab10.colors 

    for idx, (name, (t_mins, pos_ric, vel_ric)) in enumerate(segment_results_dict.items()):
        pos_np = pos_ric.detach().cpu().numpy()
        vel_np = vel_ric.detach().cpu().numpy()
        
        # Convert minutes to hours for easier reading on a 24h forecast
        t_plot_hours = t_mins.detach().cpu().numpy() / 60.0 
        
        style = {"color": colors[idx % len(colors)], "linestyle": "-", "alpha": 0.8, "linewidth": 2}

        for i in range(3):
            axes[0, i].plot(t_plot_hours, pos_np[:, i], label=name, **style)
            axes[1, i].plot(t_plot_hours, vel_np[:, i], label=name, **style)

    for i, comp in enumerate(components):
        axes[0, i].set_title(f'{comp} Error')
        axes[0, i].set_ylabel('Position Error (km)' if i == 0 else '')
        axes[1, i].set_ylabel('Velocity Error (km/s)' if i == 0 else '')
        axes[1, i].set_xlabel('Time Since Fit Epoch (Hours)')
        
        for row in range(2):
            axes[row, i].grid(True, linestyle=':', alpha=0.7)
            if i == 0 and row == 0:
                axes[row, i].legend(loc='upper left')

    plt.suptitle('6-Pass Estimator: 24-Hour RIC Residuals Post-Fit', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_median_forecast_trends(results: list[dict]):
    """Plots the median RMSE across all segments and MC iterations vs N-passes."""
    df = pl.DataFrame(results)
    
    # Aggregate to find the median RMSE for each forecast horizon
    agg_df = df.group_by("num_passes").agg([
        pl.col("1h_rmse").median().alias("1h_median"),
        pl.col("6h_rmse").median().alias("6h_median"),
        pl.col("24h_rmse").median().alias("24h_median")
    ]).sort("num_passes")
    
    # Extract arrays for plotting
    passes = agg_df["num_passes"].to_numpy()
    rmse_1h = agg_df["1h_median"].to_numpy()
    rmse_6h = agg_df["6h_median"].to_numpy()
    rmse_24h = agg_df["24h_median"].to_numpy()
    
    plt.figure(figsize=(10, 6))
    
    # Plot each horizon with distinct markers
    plt.plot(passes, rmse_1h, marker='o', linewidth=2, label="1-Hour Forecast")
    plt.plot(passes, rmse_6h, marker='s', linewidth=2, label="6-Hour Forecast")
    plt.plot(passes, rmse_24h, marker='^', linewidth=2, label="24-Hour Forecast")
    
    plt.title("Median Forecast RMSE vs. Number of Passes")
    plt.xlabel("Number of Passes")
    plt.ylabel("Median RMSE (km)")
    plt.xticks(passes)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Note: If the error drops by orders of magnitude from 0 to 1 pass, 
    # uncommenting the next line might make the 1-6 pass differences easier to read.
    # plt.yscale('log') 
    
    plt.tight_layout()
    plt.show()

import seaborn as sns

def plot_dof_forecast_trends(results: list[dict]):
    """Plots the RMSE trends vs passes, separated by DOF configuration."""
    df = pl.DataFrame(results).filter(pl.col("num_passes") >= 0).to_pandas()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    
    horizons = [
        ("1h_rmse", "1-Hour Forecast RMSE (km)"),
        ("6h_rmse", "6-Hour Forecast RMSE (km)"),
        ("24h_rmse", "24-Hour Forecast RMSE (km)")
    ]
    
    for ax, (metric_col, title) in zip(axes, horizons):
        # sns.lineplot automatically aggregates and plots the mean/median with a confidence interval band
        sns.lineplot(
            data=df, x="num_passes", y=metric_col, hue="dof_config", 
            ax=ax, marker='o', estimator='median', errorbar=('pi', 50), # 50th percentile band
            linewidth=2
        )
        ax.set_title(title)
        ax.set_xlabel("Number of Passes")
        ax.set_ylabel("Median RMSE (km)")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Log scale might be helpful to see the difference between configs clearly
        ax.set_yscale('log') 
        
    plt.tight_layout()
    plt.show()

def plot_observability_growth(data_chunks, T_mean, tle_base, center_freq):
    print("\n--- Analyzing Parameter Observability ---")
    param_names = ["n", "f", "g", "h", "k", "L", "B*"]
    passes_to_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    
    # We will just analyze the first chunk for clarity
    chunk = data_chunks[0]
    
    marginal_info_history = {p: [] for p in param_names}
    condition_number_history = []
    
    for num_passes in passes_to_test:
        active_pass_ids = chunk["pass_ids"][:num_passes]
        pass_mask = torch.isin(chunk["c"], active_pass_ids)
        
        t_active = chunk["t"][pass_mask]
        d_active_true = chunk["d_true"][pass_mask]
        c_active = chunk["c"][pass_mask]
        
        t_mean_window = float(torch.mean(t_active))
        tle_window, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
        
        # Setup pipeline strictly for the forward pass
        eval_ssv = state.MEE_SSV(init_tle=tle_window, num_measurements=len(t_active))
        eval_ssv.add_linear_bias(name="pass_freq_bias", group_indices=c_active)
        x_eval = eval_ssv.get_initial_state()
        
        t_ref_astropy = Time(t_mean_window, format="unix", scale="utc")
        station_model = system.DifferentiableStation(
            lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, 
            ref_unix=t_mean_window, 
            ref_gmst_rad=t_ref_astropy.sidereal_time('mean', 'greenwich').radian,
        )
        
        prop_eval = system.SGP4(ssv=eval_ssv)
        meas_eval = system.DopplerMeasurement(
            ssv=eval_ssv, station_model=station_model,
            freq_bias_group=eval_ssv.get_bias_group("pass_freq_bias"), time_bias_group=None
        )
        pipe_eval = system.MeasurementPipeline(propagator=prop_eval, measurement_model=meas_eval)
        
        t_since = (t_active - t_mean_window) + 0.277
        
        def forward_fn(x):
            return pipe_eval(x=x, tsince=t_since, epoch=t_mean_window, center_freq=center_freq)
            
        marg_info, evals, _ = compute_observability_metrics(
            x_state=x_eval, forward_fn=forward_fn, 
            d_obs_fixed=d_active_true, sigma_obs=20.0, param_names=param_names
        )
        
        for i, p in enumerate(param_names):
            marginal_info_history[p].append(marg_info[i])
            
        # The condition number of the normalized FIM shows if the full state is theoretically solvable
        condition_number_history.append(evals[-1] / (evals[0] + 1e-16))
        
    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for p in param_names:
        # Plotting on a log scale because n and L will vastly outpace B* and eccentricity
        axes[0].plot(passes_to_test, marginal_info_history[p], marker='o', label=p, linewidth=2)
        
    axes[0].set_yscale('log')
    axes[0].set_title("Marginal Fisher Information per Parameter")
    axes[0].set_xlabel("Number of Passes")
    axes[0].set_ylabel("Information Content (Log Scale)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='lower right')
    
    axes[1].plot(passes_to_test, condition_number_history, marker='s', color='red', linewidth=2)
    axes[1].set_yscale('log')
    axes[1].set_title("Condition Number of Normalized FIM")
    axes[1].set_xlabel("Number of Passes")
    axes[1].set_ylabel("Condition Number (Lower = More Solvable)")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def plot_ric_error_propagation(
    trajectory_dict: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    title: str = "6-Pass Estimator: 24-Hour RIC Error Propagation"
):
    """
    Plots the absolute RIC position and velocity errors over time.
    Expects trajectory_dict to map 'Configuration Name' -> (t_mins, pos_ric, vel_ric).
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    components = ['Radial', 'In-track', 'Cross-track']
    
    # Use a standard colormap to dynamically handle multiple configurations
    colors = plt.cm.tab10.colors 

    for idx, (name, (t_mins, pos_ric, vel_ric)) in enumerate(trajectory_dict.items()):
        
        # Calculate absolute error (equivalent to RMSE for a single realization)
        pos_err = np.abs(pos_ric.detach().cpu().numpy())
        vel_err = np.abs(vel_ric.detach().cpu().numpy())
        
        # t_mins is already relative to the fit epoch, just convert to hours
        t_plot_hours = t_mins.detach().cpu().numpy() / 60.0 
        
        style = {"color": colors[idx % len(colors)], "linestyle": "-", "alpha": 0.8, "linewidth": 2}

        for i in range(3):
            axes[0, i].plot(t_plot_hours, pos_err[:, i], label=name, **style)
            axes[1, i].plot(t_plot_hours, vel_err[:, i], label=name, **style)

    for i, comp in enumerate(components):
        axes[0, i].set_title(f'{comp} Error Magnitude')
        axes[0, i].set_ylabel('Position Error (km)' if i == 0 else '')
        axes[1, i].set_ylabel('Velocity Error (km/s)' if i == 0 else '')
        axes[1, i].set_xlabel('Time Since Fit Epoch (Hours)')
        
        for row in range(2):
            axes[row, i].grid(True, linestyle='--', alpha=0.6)
            if i == 0 and row == 0:
                axes[row, i].legend(loc='upper left')
                
            # Optional: Uncomment if error grows by orders of magnitude and becomes hard to read
            # axes[row, i].set_yscale('log')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_pca_information_space(data_chunks, tle_base, center_freq, num_passes=3):
    """
    Performs a PCA-equivalent eigendecomposition on the normalized Fisher 
    Information Matrix to show the dominance of specific MEE parameters.
    """
    param_names = ["n", "f", "g", "L"]
    chunk = data_chunks[0]
    
    # 1. Setup the 3-pass data
    active_pass_ids = chunk["pass_ids"][:num_passes]
    pass_mask = torch.isin(chunk["c"], active_pass_ids)
    
    t_active = chunk["t"][pass_mask]
    d_active_true = chunk["d_true"][pass_mask]
    c_active = chunk["c"][pass_mask]
    
    t_mean_window = float(torch.mean(t_active))
    tle_window, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
    
    # 2. Initialize SSV and Forward Model
    eval_ssv = state.MEE_SSV(init_tle=tle_window, num_measurements=len(t_active))
    eval_ssv.add_linear_bias(name="pass_freq_bias", group_indices=c_active)
    x_eval = eval_ssv.get_initial_state()
    
    t_ref_astropy = Time(t_mean_window, format="unix", scale="utc")
    station_model = system.DifferentiableStation(
        lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, 
        ref_unix=t_mean_window, 
        ref_gmst_rad=t_ref_astropy.sidereal_time('mean', 'greenwich').radian,
        # device=device
    )
    
    prop_eval = system.SGP4(ssv=eval_ssv)
    meas_eval = system.DopplerMeasurement(
        ssv=eval_ssv, station_model=station_model,
        freq_bias_group=eval_ssv.get_bias_group("pass_freq_bias"), time_bias_group=None
    )
    pipe_eval = system.MeasurementPipeline(propagator=prop_eval, measurement_model=meas_eval)
    t_since = (t_active - t_mean_window) + 0.277
    
    def forward_fn(x):
        return pipe_eval(x=x, tsince=t_since, epoch=t_mean_window, center_freq=center_freq)
        
    # 3. Compute Observability (Eigendecomposition of Normalized FIM)
    _, evals, evecs = compute_observability_metrics(
        x_state=x_eval, forward_fn=forward_fn, 
        d_obs_fixed=d_active_true, sigma_obs=20.0, param_names=param_names
    )
    
    # Sort eigenvalues/eigenvectors in descending order (Largest Information -> Smallest)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # Calculate Explained Information Ratio (PCA equivalent)
    explained_info_ratio = evals / np.sum(evals)
    cumulative_info = np.cumsum(explained_info_ratio)
    
    # Calculate the fractional contribution of each parameter to each Principal Component
    # We square the eigenvectors to get the variance/information magnitude
    pc_composition = evecs**2
    
    # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot A: Scree Plot (Explained Information)
    axes[0].bar(range(1, len(evals)+1), explained_info_ratio * 100, alpha=0.7, label="Individual PC")
    axes[0].plot(range(1, len(evals)+1), cumulative_info * 100, marker='o', color='red', label="Cumulative")
    axes[0].set_title(f"PCA of Information Space ({num_passes} Passes)")
    axes[0].set_xlabel("Principal Component (PC)")
    axes[0].set_ylabel("Explained Information (%)")
    axes[0].set_xticks(range(1, len(evals)+1))
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot B: Heatmap of PC Composition
    sns.heatmap(
        pc_composition.T, 
        annot=True, 
        cmap="YlGnBu", 
        xticklabels=param_names,
        yticklabels=[f"PC {i+1}" for i in range(len(evals))],
        ax=axes[1],
        fmt=".2f"
    )
    axes[1].set_title("Parameter Composition of Principal Components")
    axes[1].set_xlabel("Modified Equinoctial Elements")
    axes[1].set_ylabel("Principal Components")
    
    plt.tight_layout()
    plt.show()

from torch.func import jacfwd

def plot_parameter_correlation_evolution(data_chunks, tle_base, center_freq, passes_to_plot=[1, 3, 8]):
    """
    Plots the parameter correlation matrix for different numbers of passes
    to show how parameter coupling changes as more data is introduced.
    """
    param_names = ["n", "L"]#, "g", "h", "k", "L", "B*"]
    num_params = len(param_names)
    chunk = data_chunks[0]
    
    # Setup the figure for multiple subplots
    fig, axes = plt.subplots(1, len(passes_to_plot), figsize=(6 * len(passes_to_plot), 5.5))
    
    for idx, num_passes in enumerate(passes_to_plot):
        # 1. Setup the data for N passes
        active_pass_ids = chunk["pass_ids"][:num_passes]
        pass_mask = torch.isin(chunk["c"], active_pass_ids)
        
        t_active = chunk["t"][pass_mask]
        d_active_true = chunk["d_true"][pass_mask]
        c_active = chunk["c"][pass_mask]
        
        t_mean_window = float(torch.mean(t_active))
        tle_window, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
        
        # 2. Initialize SSV and Forward Model
        eval_ssv = state.MEE_SSV(init_tle=tle_window, num_measurements=len(t_active))
        eval_ssv.add_linear_bias(name="pass_freq_bias", group_indices=c_active)
        x_eval = eval_ssv.get_initial_state()
        
        t_ref_astropy = Time(t_mean_window, format="unix", scale="utc")
        station_model = system.DifferentiableStation(
            lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, 
            ref_unix=t_mean_window, 
            ref_gmst_rad=t_ref_astropy.sidereal_time('mean', 'greenwich').radian,
            # device=device # Uncomment if running strictly on GPU
        )
        
        prop_eval = system.SGP4(ssv=eval_ssv)
        meas_eval = system.DopplerMeasurement(
            ssv=eval_ssv, station_model=station_model,
            freq_bias_group=eval_ssv.get_bias_group("pass_freq_bias"), time_bias_group=None
        )
        pipe_eval = system.MeasurementPipeline(propagator=prop_eval, measurement_model=meas_eval)
        t_since = (t_active - t_mean_window) + 0.277
        
        # 3. Compute Jacobian directly
        def res_fn(x):
            return pipe_eval(x=x, tsince=t_since, epoch=t_mean_window, center_freq=center_freq) - d_active_true
            
        H_total = jacfwd(res_fn)(x_eval)
        
        # Isolate the orbital parameters (ignore bias parameters)
        H_orb = H_total[:, :num_params]
        
        # 4. Compute FIM, Covariance, and Correlation
        sigma_obs = 20.0
        W = 1.0 / (sigma_obs**2)
        FIM = H_orb.T @ (H_orb * W)
        
        # Use pseudo-inverse to handle singularity at low pass counts
        Cov = torch.linalg.pinv(FIM)
        
        # Extract standard deviations (sqrt of diagonal elements)
        std_dev = torch.sqrt(torch.abs(torch.diag(Cov)))
        
        # Compute Correlation Matrix: C_ij = Cov_ij / (std_i * std_j)
        # Add a tiny epsilon to prevent division by zero for totally unobservable states
        outer_std = torch.outer(std_dev, std_dev) + 1e-16
        Corr = Cov / outer_std
        
        # Convert to numpy for Seaborn
        Corr_np = Corr.detach().cpu().numpy()
        
        # Calculate the Distance to Orthogonality (Frobenius norm of off-diagonals)
        # identity = np.eye(num_params)
        frob_norm_off_diag = np.linalg.det(Corr_np)
        
        # 5. Plotting
        ax = axes[idx] if len(passes_to_plot) > 1 else axes
        sns.heatmap(
            Corr_np, 
            annot=True, 
            cmap="coolwarm", 
            vmin=-1, vmax=1, 
            xticklabels=param_names,
            yticklabels=param_names,
            fmt=".2f",
            ax=ax,
            square=True,
            cbar_kws={"shrink": .8}
        )
        # Update the title to include the metric
        ax.set_title(f"{num_passes} Passes | $\|det(C)\|_F = {frob_norm_off_diag:.2f}$")
        ax.set_xlabel("Modified Equinoctial Elements")
        if idx == 0:
            ax.set_ylabel("Modified Equinoctial Elements")

    plt.tight_layout()
    plt.show()

import polars as pl
import pandas as pd

def generate_rmse_statistics_table(results: list[dict], horizons=["1h", "24h"]):
    """
    Generates a formatted table showing the Mean ± Std Dev of RMSE 
    for different DOF configurations and number of passes.
    """
    # 1. Load results into Polars
    df = pl.DataFrame(results)
    
    # Filter out baseline runs if you only want to compare active OD configs
    df = df.filter(pl.col("num_passes") > 0)
    
    # 2. Build the aggregation expressions dynamically based on requested horizons
    agg_exprs = []
    for h in horizons:
        col_name = f"{h}_rmse"
        agg_exprs.extend([
            pl.col(col_name).mean().alias(f"{h}_mean"),
            pl.col(col_name).std().alias(f"{h}_std")
        ])
        
    # Group by passes and configuration, aggregating across all chunks and MC iterations
    agg_df = df.group_by(["num_passes", "dof_config"]).agg(agg_exprs).sort(["num_passes", "dof_config"])
    
    # 3. Convert to Pandas for string formatting and multi-index pivoting
    pdf = agg_df.to_pandas()
    
    # Create the 'Mean ± Std' strings
    for h in horizons:
        pdf[f"{h} Forecast (km)"] = pdf.apply(
            lambda row: f"{row[f'{h}_mean']:.2f} ± {row[f'{h}_std']:.2f}", axis=1
        )
    
    # Keep only the formatted columns
    cols_to_keep = ["num_passes", "dof_config"] + [f"{h} Forecast (km)" for h in horizons]
    pdf = pdf[cols_to_keep]
    
    # 4. Pivot the table: Rows = passes, Columns = Configurations and Horizons
    pivot_df = pdf.pivot(
        index="num_passes", 
        columns="dof_config", 
        values=[f"{h} Forecast (km)" for h in horizons]
    )
    
    # Flatten the MultiIndex columns for a cleaner look
    # E.g., ('1h Forecast (km)', '2-DOF (n, L)') -> '2-DOF (n, L) | 1h Forecast (km)'
    pivot_df.columns = [f"{config} | {metric}" for metric, config in pivot_df.columns]
    
    # Ensure rows are sorted by pass count
    pivot_df = pivot_df.sort_index()
    
    return pivot_df

# Example execution:
# summary_table = generate_rmse_statistics_table(experiment_results, horizons=["1h", "24h"])
# print(summary_table)