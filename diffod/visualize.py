from astropy import conf
from astropy.units import hh
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

def plot_passes_correlation_comparison(data_chunks, tle_base, center_freq):
    """
    Plots the parameter correlation matrix for the full 7-DOF model
    across different amounts of data (1, 3, and 6 passes).
    """
    # 1. Define the pass counts to compare and lock in the 7-DOF configuration
    pass_counts = [1, 3, 6]
    param_names = ["n", "L", "f", "g", "k", "h", "B*"]
    col_indices = [0, 5, 1, 2, 4, 3, 6]
    
    chunk = data_chunks[0]
    
    # Setup the figure for multiple subplots
    fig, axes = plt.subplots(1, len(pass_counts), figsize=(6 * len(pass_counts), 5.5))
    
    # 2. Loop through the different pass configurations
    for idx, num_passes in enumerate(pass_counts):
        
        # Setup the data for the specific number of passes
        active_pass_ids = chunk["pass_ids"][:num_passes]
        pass_mask = torch.isin(chunk["c"], active_pass_ids)
        
        t_active = chunk["t"][pass_mask]
        d_active_true = chunk["d_true"][pass_mask]
        c_active = chunk["c"][pass_mask]
        
        t_mean_window = float(torch.mean(t_active))
        tle_window, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
        
        # 3. Base Evaluation Setup (must be recreated per pass count since t_active changes)
        eval_ssv = state.MEE_SSV(init_tle=tle_window, num_measurements=len(t_active))
        eval_ssv.add_linear_bias(name="pass_freq_bias", group_indices=c_active)
        x_eval = eval_ssv.get_initial_state()
        
        t_ref_astropy = Time(t_mean_window, format="unix", scale="utc")
        station_model = system.DifferentiableStation(
            lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, 
            ref_unix=t_mean_window, 
            ref_gmst_rad=t_ref_astropy.sidereal_time('mean', 'greenwich').radian
        )
        
        prop_eval = system.SGP4(ssv=eval_ssv, use_pretrained_model=True)
        meas_eval = system.DopplerMeasurement(
            ssv=eval_ssv, station_model=station_model,
            freq_bias_group=eval_ssv.get_bias_group("pass_freq_bias"), time_bias_group=None
        )
        pipe_eval = system.MeasurementPipeline(propagator=prop_eval, measurement_model=meas_eval)
        t_since = (t_active - t_mean_window) + 0.277
        
        # 4. Compute Full Jacobian for this specific chunk of data
        def res_fn(x):
            return pipe_eval(x=x, tsince=t_since, epoch=t_mean_window, center_freq=center_freq) - d_active_true
            
        H_total = jacfwd(res_fn)(x_eval)
        
        # Isolate only the specified 7 parameters
        H_orb = H_total[:, col_indices]
        
        # --- Scale the Jacobian for numerical stability ---
        col_scales = torch.max(torch.abs(H_orb), dim=0)[0]
        col_scales[col_scales == 0] = 1.0  
        H_scaled = H_orb / col_scales
        
        # Compute FIM with the scaled Jacobian
        sigma_obs = 20.0
        W = 1.0 / (sigma_obs**2)
        FIM_scaled = H_scaled.T @ (H_scaled * W)
        
        # Use pseudo-inverse on the well-conditioned matrix
        Cov_scaled = torch.linalg.pinv(FIM_scaled)
        
        # Compute standard deviations
        std_dev = torch.sqrt(torch.abs(torch.diag(Cov_scaled)))
        outer_std = torch.outer(std_dev, std_dev)

        # FIM_numpy = FIM_scaled.detach().cpu().numpy()
        Corr_np = np.linalg.inv(Cov_scaled.detach().cpu().numpy())
        
        # Compute Correlation
        Corr = Cov_scaled / (outer_std + 1e-12)
        Corr.fill_diagonal_(1.0) # Force exact 1.0 on diagonals
        
        Corr_np = Corr.detach().cpu().numpy()
        
        # Calculate the Determinant (Generalized Variance)
        det_C = np.linalg.det(Corr_np)
        
        # 5. Plotting
        ax = axes[idx] if len(pass_counts) > 1 else axes
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
        
        ax.set_title(f"7-DOF ({num_passes} Passes)\n$\\det(C) = {det_C:.2e}$")
        ax.set_xlabel("Modified Equinoctial Elements")
        if idx == 0:
            ax.set_ylabel("Modified Equinoctial Elements")

    plt.tight_layout()
    plt.show()

import numpy as np
import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# from astropy.time import Time
# from torch.autograd.functional import jacfwd

# (Assuming state, system, dsgp4, and unix_to_mjd are imported)

def plot_passes_mi_comparison(data_chunks, tle_base, center_freq):
    """
    Plots the Mutual Information matrix for the full 7-DOF model
    across different amounts of data (1, 3, and 6 passes).
    """
    # 1. Define the pass counts and configurations
    pass_counts = [1, 3, 6]
    param_names = ["n", "L", "f", "g", "k", "h", "B*"]
    col_indices = [0, 5, 1, 2, 4, 3, 6]
    
    chunk = data_chunks[0]
    
    fig, axes = plt.subplots(1, len(pass_counts), figsize=(6 * len(pass_counts), 5.5))
    
    # 2. Loop through the different pass configurations
    for idx, num_passes in enumerate(pass_counts):
        
        active_pass_ids = chunk["pass_ids"][:num_passes]
        pass_mask = torch.isin(chunk["c"], active_pass_ids)
        
        t_active = chunk["t"][pass_mask]
        d_active_true = chunk["d_true"][pass_mask]
        c_active = chunk["c"][pass_mask]
        
        t_mean_window = float(torch.mean(t_active))
        tle_window, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
        
        # 3. Base Evaluation Setup
        eval_ssv = state.MEE_SSV(init_tle=tle_window, num_measurements=len(t_active))
        eval_ssv.add_linear_bias(name="pass_freq_bias", group_indices=c_active)
        x_eval = eval_ssv.get_initial_state()
        
        t_ref_astropy = Time(t_mean_window, format="unix", scale="utc")
        station_model = system.DifferentiableStation(
            lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, 
            ref_unix=t_mean_window, 
            ref_gmst_rad=t_ref_astropy.sidereal_time('mean', 'greenwich').radian
        )
        
        prop_eval = system.SGP4(ssv=eval_ssv, use_pretrained_model=True)
        meas_eval = system.DopplerMeasurement(
            ssv=eval_ssv, station_model=station_model,
            freq_bias_group=eval_ssv.get_bias_group("pass_freq_bias"), time_bias_group=None
        )
        pipe_eval = system.MeasurementPipeline(propagator=prop_eval, measurement_model=meas_eval)
        t_since = (t_active - t_mean_window) + 0.277
        
        # 4. Compute Full Jacobian and scale it
        def res_fn(x):
            return pipe_eval(x=x, tsince=t_since, epoch=t_mean_window, center_freq=center_freq) - d_active_true
            
        H_total = jacfwd(res_fn)(x_eval)
        H_orb = H_total[:, col_indices]
        
        col_scales = torch.max(torch.abs(H_orb), dim=0)[0]
        col_scales[col_scales == 0] = 1.0  
        H_scaled = H_orb / col_scales
        
        # Compute FIM and Covariance
        sigma_obs = 20.0
        W = 1.0 / (sigma_obs**2)
        FIM_scaled = H_scaled.T @ (H_scaled * W)
        Cov_scaled = torch.linalg.pinv(FIM_scaled)
        
        # Compute Correlation
        std_dev = torch.sqrt(torch.abs(torch.diag(Cov_scaled)))
        outer_std = torch.outer(std_dev, std_dev)
        Corr = Cov_scaled / (outer_std + 1e-12)
        Corr.fill_diagonal_(1.0) 
        
        Corr_np = Corr.detach().cpu().numpy()
        
        # --- NEW: Compute Mutual Information ---
        # Clip squared correlation slightly below 1 to prevent log(0) errors on off-diagonals
        rho_sq = np.clip(Corr_np**2, 0, 0.99999) 
        MI_np = -0.5 * np.log(1 - rho_sq)
        
        # Set diagonal to NaN so 'infinity' doesn't ruin the heatmap color mapping
        np.fill_diagonal(MI_np, np.nan)
        
        det_FIM = np.linalg.det(FIM_scaled.detach().cpu().numpy())
        
        # 5. Plotting
        ax = axes[idx] if len(pass_counts) > 1 else axes
        sns.heatmap(
            MI_np, 
            annot=True, 
            cmap="Reds",    # Changed to a sequential colormap
            vmin=0,         # MI is strictly >= 0
            vmax=5,         # Cap the colorbar at 5 nats to highlight differences
            xticklabels=param_names,
            yticklabels=param_names,
            fmt=".2f",
            ax=ax,
            square=True,
            cbar_kws={"shrink": .8, "label": "Mutual Info (nats)"}
        )
        
        ax.set_title(f"Mutual Info ({num_passes} Passes)\n$\\det(FIM) = {det_FIM:.2e}$")
        ax.set_xlabel("Modified Equinoctial Elements")
        if idx == 0:
            ax.set_ylabel("Modified Equinoctial Elements")

    plt.tight_layout()
    plt.show()

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
        ax.legend(title="DOF Configuration")
        ax.set_title(title)
        ax.set_xlabel("Number of Passes")
        ax.set_ylabel("Median RMSE (km)")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Log scale might be helpful to see the difference between configs clearly
        ax.set_yscale('log') 
        
    plt.tight_layout()
    plt.savefig("forecast_trends.png", dpi=500)
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

# import numpy as np
# import torch
# import matplotlib.subplots as plt
# import seaborn as sns
# from astropy.time import Time
# from torch.func import jacfwd
# import dsgp4

def plot_doppler_variance_pca(data_chunks, tle_base, center_freq, passes_to_plot=[1, 3, 6]):
    """
    Performs a PCA on the Doppler measurement Jacobian to show
    the absolute signal variance (in Hz^2) explained by the MEE parameters.
    """
    param_names = ["n", "f", "L", "g"]
    chunk = data_chunks[0]
    
    fig, axes = plt.subplots(2, len(passes_to_plot), figsize=(6 * len(passes_to_plot), 10))
    
    for idx, num_passes in enumerate(passes_to_plot):
        active_pass_ids = chunk["pass_ids"][:num_passes]
        pass_mask = torch.isin(chunk["c"], active_pass_ids)
        
        t_active = chunk["t"][pass_mask]
        d_active_true = chunk["d_true"][pass_mask]
        c_active = chunk["c"][pass_mask]
        
        t_mean_window = float(torch.mean(t_active))
        tle_window, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
        
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
        
        # Compute Full Jacobian
        def res_fn(x):
            return pipe_eval(x=x, tsince=t_since, epoch=t_mean_window, center_freq=center_freq)
            
        H_total = jacfwd(res_fn)(x_eval)
        H_orb = H_total[:, :len(param_names)]
        
        # Scale by a standard prior variance to ensure units are physically balanced
        # (e.g., simulating a baseline 1e-4 physical perturbation to the state)
        state_perturbation_variance = (1e-4)**2
        
        # Compute the Absolute Signal Variance Matrix (in Hz^2)
        Signal_Variance_Matrix = (H_orb.T @ H_orb) * state_perturbation_variance
        
        evals, evecs = torch.linalg.eigh(Signal_Variance_Matrix)
        
        # Sort in descending order
        evals = evals.detach().cpu().numpy()
        evecs = evecs.detach().cpu().numpy()
        sort_idx = np.argsort(evals)[::-1]
        evals = evals[sort_idx]
        evecs = evecs[:, sort_idx]
        
        # Calculate the PC composition
        pc_composition = evecs**2
        
        # Plot A: Absolute Variance (Scree Plot)
        ax_bar = axes[0, idx]
        ax_bar.bar(range(1, len(evals)+1), evals, alpha=0.8, color='steelblue')
        ax_bar.set_yscale('log')
        ax_bar.set_title(f"Doppler Signal Variance ({num_passes} Passes)\nTotal Var: {np.sum(evals):.2e} $Hz^2$")
        ax_bar.set_xlabel("Principal Component (PC)")
        ax_bar.set_ylabel("Absolute Explained Variance ($Hz^2$)")
        ax_bar.set_xticks(range(1, len(evals)+1))
        ax_bar.grid(True, linestyle='--', alpha=0.6)
        
        # Plot B: Heatmap of PC Composition
        ax_heat = axes[1, idx]
        sns.heatmap(
            pc_composition.T, 
            annot=True, 
            cmap="YlGnBu", 
            xticklabels=param_names,
            yticklabels=[f"PC {i+1}" for i in range(len(evals))],
            ax=ax_heat,
            fmt=".2f",
            cbar_kws={"shrink": .8}
        )
        ax_heat.set_title("Parameter Composition of PCs")
        ax_heat.set_xlabel("Modified Equinoctial Elements")

    plt.tight_layout()
    plt.show()

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# from astropy.time import Time
from torch.func import jacfwd
# import dsgp4

def plot_total_parameter_variance(data_chunks, tle_base, center_freq, num_passes=3):
    """
    Computes the Total Parameter Importance by weighting the SVD parameter 
    composition by the variance explained by each principal component.
    """
    param_names = ["n", "L"]#, "L", "g", "f", "h", "k"]
    chunk = data_chunks[0]
    
    # 1. Setup Data
    active_pass_ids = chunk["pass_ids"][:num_passes]
    pass_mask = torch.isin(chunk["c"], active_pass_ids)
    
    t_active = chunk["t"][pass_mask]
    d_active_true = chunk["d_true"][pass_mask]
    c_active = chunk["c"][pass_mask]
    
    t_mean_window = float(torch.mean(t_active))
    tle_window, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
    
    # 2. Setup Forward Model
    eval_ssv = state.MEE_SSV(init_tle=tle_window, num_measurements=len(t_active))
    eval_ssv.add_linear_bias(name="pass_freq_bias", group_indices=c_active)
    x_eval = eval_ssv.get_initial_state()
    
    # ... [Initialize station_model, prop_eval, meas_eval, pipe_eval as before] ...
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
    
    # 3. Compute Initial Residuals and Jacobian
    with torch.no_grad():
        d_pred_initial = pipe_eval(x=x_eval, tsince=t_since, epoch=t_mean_window, center_freq=center_freq)
        
    residuals = d_active_true - d_pred_initial
    
    def forward_fn(x):
        return pipe_eval(x=x, tsince=t_since, epoch=t_mean_window, center_freq=center_freq)
        
    H_total = jacfwd(forward_fn)(x_eval)
    H_orb = H_total[:, :len(param_names)] 
    
    # 4. Normalize the Jacobian
    col_norms = torch.norm(H_orb, dim=0)
    H_norm = H_orb / (col_norms + 1e-16)
    
    # 5. SVD on Measurement Space
    U, S, Vh = torch.linalg.svd(H_norm, full_matrices=False)
    V = Vh.T
    
    # 6. Project residuals to find Variance Explained per PC
    c = U.T @ residuals
    total_variance = torch.sum(residuals**2).item()
    variance_explained = (c**2).detach().cpu().numpy()
    pct_explained = (variance_explained / total_variance) * 100.0
    
    pct_unexplained = max(0.0, 100.0 - np.sum(pct_explained))
    
    # 7. Calculate Total Parameter Attribution
    # Matrix multiply the composition (V^2) by the variance explained vector
    pc_composition = (V**2).detach().cpu().numpy()
    param_attribution = np.sum(pc_composition * pct_explained, axis=1)
    
    # 8. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot A: Cumulative Variance Explained (Effective Dimensionality)
    cumulative_variance = np.cumsum(pct_explained)
    axes[0].plot(range(1, len(param_names)+1), cumulative_variance, marker='o', linewidth=2, color='red')
    axes[0].bar(range(1, len(param_names)+1), pct_explained, alpha=0.6, label="Individual PC")
    axes[0].axhline(95.0, color='black', linestyle='--', label="95% Threshold")
    
    axes[0].set_title(f"Effective System Dimensionality ({num_passes} Passes)")
    axes[0].set_xlabel("Principal Component (Ranked)")
    axes[0].set_ylabel("Cumulative Variance Explained (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot B: Total Parameter Attribution
    # We add the "Unexplained" variance as the final bar to ensure everything sums to 100%
    plot_labels = param_names + ["Unmodeled/Noise"]
    plot_values = list(param_attribution) + [pct_unexplained]
    colors = ['#2ca02c']*len(param_names) + ['#d62728']
    
    bars = axes[1].bar(plot_labels, plot_values, color=colors, alpha=0.8)
    axes[1].set_title("Total Parameter Attribution to System Variance")
    axes[1].set_ylabel("Total Explained Variance (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        if yval > 0.5: # Only label visible bars
            axes[1].text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10)

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

def plot_ric_state_variance_attribution(t_gps, r_gps, v_gps, tle_base, epoch_unix):
    """
    Performs SVD on the mapping from MEE parameters to the physical RIC state space
    to prove that n and L strictly govern the dominant physical trajectory errors.
    """
    param_names = ["n", "f", "g", "h", "k", "L", "B*"]
    
    # 1. Setup a 24-hour window for trajectory analysis
    mask = (t_gps >= epoch_unix) & (t_gps <= epoch_unix + 86400)
    t_eval = t_gps[mask]
    r_eval = r_gps[mask].to(torch.float64)
    v_eval = v_gps[mask].to(torch.float64)
    
    # 2. Setup SGP4 Forward Model
    init_tle, _ = dsgp4.newton_method(tle_base, unix_to_mjd(epoch_unix))
    ssv = state.MEE_SSV(init_tle=init_tle, num_measurements=1, fit_bstar=False)
    prop = system.SGP4(ssv=ssv)
    x_eval = ssv.get_initial_state()
    
    t_since_mins = (t_eval - epoch_unix) / 60.0
    
    # 3. Create a differentiable function that outputs RIC Position
    def ric_forward_fn(x):
        r_calc, v_calc = prop(x=x, tsince=t_since_mins)
        
        # Construct RIC Basis Vectors using GPS as truth
        R_hat = r_eval / torch.norm(r_eval, dim=1, keepdim=True)
        W_vec = torch.cross(r_eval, v_eval, dim=1)
        W_hat = W_vec / torch.norm(W_vec, dim=1, keepdim=True)
        S_hat = torch.cross(W_hat, R_hat, dim=1)
        
        # Project calculated position onto RIC
        pos_res_R = (r_calc * R_hat).sum(dim=1).unsqueeze(1)
        pos_res_I = (r_calc * S_hat).sum(dim=1).unsqueeze(1)
        pos_res_C = (r_calc * W_hat).sum(dim=1).unsqueeze(1)
        
        return torch.cat([pos_res_R, pos_res_I, pos_res_C], dim=1)
        
    # 4. Compute Jacobian of RIC Position w.r.t MEE Parameters
    # Shape: (N_times, 3, N_params). We flatten the spatial/time dims for SVD
    J_ric = jacfwd(ric_forward_fn)(x_eval)
    J_ric_flat = J_ric.view(-1, J_ric.shape[-1])[:, :len(param_names)]
    
    # Normalize columns to compare dimensionless parameters
    col_norms = torch.norm(J_ric_flat, dim=0)
    J_norm = J_ric_flat / (col_norms + 1e-16)
    
    # 5. Calculate True Physical Residuals
    with torch.no_grad():
        r_prior = ric_forward_fn(x_eval)
        # Truth projected into its own RIC frame is just [R, 0, 0] in ideal conditions,
        # but the delta is simply the r_prior relative to the origin of the RIC frame
        true_residuals_flat = r_prior.view(-1) 
        
    # 6. Perform SVD on the State Mapping
    U, S, Vh = torch.linalg.svd(J_norm, full_matrices=False)
    V = Vh.T
    
    c = U.T @ true_residuals_flat
    total_variance = torch.sum(true_residuals_flat**2).item()
    pct_explained = ((c**2).detach().cpu().numpy() / total_variance) * 100.0
    
    pc_composition = (V**2).detach().cpu().numpy()
    param_attribution = np.sum(pc_composition * pct_explained[:, None], axis=0)
    
    # 7. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot A: True Trajectory Variance by Axis
    r_err = r_prior[:, 0].detach().cpu().numpy()
    i_err = r_prior[:, 1].detach().cpu().numpy()
    c_err = r_prior[:, 2].detach().cpu().numpy()
    
    var_R = np.var(r_err)
    var_I = np.var(i_err)
    var_C = np.var(c_err)
    total_var_ric = var_R + var_I + var_C
    
    axes[0].bar(['Radial', 'In-Track', 'Cross-Track'], 
                [var_R/total_var_ric * 100, var_I/total_var_ric * 100, var_C/total_var_ric * 100], 
                color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    axes[0].set_title("True Physical Trajectory Variance (24h Forecast)")
    axes[0].set_ylabel("Percentage of Total Spatial Variance (%)")
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Plot B: Parameter Attribution to Physical State
    axes[1].bar(param_names, param_attribution, color='#9467bd', alpha=0.8)
    axes[1].set_title("MEE Parameter Attribution to Physical State Error")
    axes[1].set_ylabel("Variance Explained (%)")
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.func import jacfwd
from astropy.time import Time
import dsgp4

def plot_estimator_utility_space(data_chunks, t_gps, r_gps, v_gps, tle_base, center_freq, epoch_unix, num_passes=3):
    """
    Computes both Measurement (Doppler) Attribution and Physical (RIC) Attribution,
    mapping parameters into a 2D utility space to justify optimal parameter selection.
    """
    param_names = ["n", "f", "g", "h", "k", "L", "B*"]
    chunk = data_chunks[0]
    
    # ---------------------------------------------------------
    # PART 1: DOPPLER MEASUREMENT PCA (Observability)
    # ---------------------------------------------------------
    active_pass_ids = chunk["pass_ids"][:num_passes]
    pass_mask = torch.isin(chunk["c"], active_pass_ids)
    
    t_active = chunk["t"][pass_mask]
    d_active_true = chunk["d_true"][pass_mask]
    c_active = chunk["c"][pass_mask]
    
    t_mean_window = float(torch.mean(t_active))
    tle_window, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
    
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
    t_since_dopp = (t_active - t_mean_window) + 0.277
    
    with torch.no_grad():
        d_pred = pipe_eval(x=x_eval, tsince=t_since_dopp, epoch=t_mean_window, center_freq=center_freq)
    dopp_res = d_active_true - d_pred
    
    def dopp_forward(x): return pipe_eval(x=x, tsince=t_since_dopp, epoch=t_mean_window, center_freq=center_freq)
    J_dopp = jacfwd(dopp_forward)(x_eval)[:, :len(param_names)]
    J_dopp_norm = J_dopp / (torch.norm(J_dopp, dim=0) + 1e-16)
    
    U_d, S_d, Vh_d = torch.linalg.svd(J_dopp_norm, full_matrices=False)
    c_d = U_d.T @ dopp_res
    pct_explained_dopp = ((c_d**2).detach().cpu().numpy() / torch.sum(dopp_res**2).item()) * 100.0
    dopp_attribution = np.sum((Vh_d.T**2).detach().cpu().numpy() * pct_explained_dopp, axis=1)
    
    # ---------------------------------------------------------
    # PART 2: PHYSICAL STATE PCA (Explainability)
    # ---------------------------------------------------------
    mask_gps = (t_gps >= epoch_unix) & (t_gps <= epoch_unix + 86400)
    t_eval = t_gps[mask_gps]
    r_eval = r_gps[mask_gps].to(torch.float64)
    v_eval = v_gps[mask_gps].to(torch.float64)
    t_since_gps = (t_eval - epoch_unix) / 60.0
    
    def ric_forward(x):
        r_calc, v_calc = prop_eval(x=x, tsince=t_since_gps)
        R_hat = r_eval / torch.norm(r_eval, dim=1, keepdim=True)
        W_vec = torch.cross(r_eval, v_eval, dim=1)
        W_hat = W_vec / torch.norm(W_vec, dim=1, keepdim=True)
        S_hat = torch.cross(W_hat, R_hat, dim=1)
        return torch.cat([
            (r_calc * R_hat).sum(dim=1).unsqueeze(1),
            (r_calc * S_hat).sum(dim=1).unsqueeze(1),
            (r_calc * W_hat).sum(dim=1).unsqueeze(1)
        ], dim=1)
        
    J_ric = jacfwd(ric_forward)(x_eval).view(-1, len(x_eval))[:, :len(param_names)]
    J_ric_norm = J_ric / (torch.norm(J_ric, dim=0) + 1e-16)
    
    with torch.no_grad():
        r_prior = ric_forward(x_eval).view(-1)
        
    U_r, S_r, Vh_r = torch.linalg.svd(J_ric_norm, full_matrices=False)
    c_r = U_r.T @ r_prior
    pct_explained_ric = ((c_r**2).detach().cpu().numpy() / torch.sum(r_prior**2).item()) * 100.0
    ric_attribution = np.sum((Vh_r.T**2).detach().cpu().numpy() * pct_explained_ric[:, None], axis=0)

    # ---------------------------------------------------------
    # PART 3: ESTIMATION UTILITY INDEX (EUI)
    # ---------------------------------------------------------
    # EUI is the geometric mean (or product) of Measurement Visibility and Physical Impact
    # Normalized so the highest value is 1.0 (or 100%)
    utility_index = np.sqrt(dopp_attribution * ric_attribution)
    utility_index = (utility_index / np.max(utility_index)) * 100.0
    
    # ---------------------------------------------------------
    # 4. PLOTTING
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot A: 2D Pareto Front Space
    axes[0].scatter(dopp_attribution, ric_attribution, color='blue', s=100, zorder=5)
    
    # Add parameter labels
    for i, txt in enumerate(param_names):
        axes[0].annotate(txt, (dopp_attribution[i], ric_attribution[i]), 
                         xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
        
    # Draw quadrants based on medians or hard thresholds to show the "Sweet Spot"
    axes[0].axvline(np.median(dopp_attribution), color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(np.median(ric_attribution), color='gray', linestyle='--', alpha=0.5)
    
    # Shade the "Optimal Estimation Region" (Top Right Quadrant)
    axes[0].axvspan(np.median(dopp_attribution), max(dopp_attribution)*1.1, 
                    ymin=np.median(ric_attribution)/max(ric_attribution)*0.9, ymax=1, 
                    color='green', alpha=0.1, label='Optimal Estimation Region')

    axes[0].set_title(f"Observability-Explainability Space ({num_passes} Passes)")
    axes[0].set_xlabel("Measurement Observability (Doppler Variance Explained %)")
    axes[0].set_ylabel("Physical State Explainability (RIC Variance Explained %)")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper left')
    
    # Plot B: Estimation Utility Index
    # Sort for cleaner presentation
    sort_idx = np.argsort(utility_index)[::-1]
    sorted_params = [param_names[i] for i in sort_idx]
    sorted_utility = utility_index[sort_idx]
    
    colors = ['#2ca02c' if p in ['n', 'L'] else '#1f77b4' for p in sorted_params]
    
    bars = axes[1].bar(sorted_params, sorted_utility, color=colors, alpha=0.8)
    axes[1].set_title("Estimation Utility Index (Cross-Domain Alignment)")
    axes[1].set_ylabel("Relative Utility Score (%)")
    axes[1].set_ylim(0, 110)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

# Execution:
# plot_estimator_utility_space(data_chunks, t_gps, r_gps, v_gps, tle_base, center_freq, epoch_unix, num_passes=3)

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import jacfwd
import dsgp4

def plot_observability_state_alignment(data_chunks, t_gps, r_gps, v_gps, tle_base, center_freq, epoch_unix, num_passes=3):
    """
    Computes the Measurement PC1 and State PC1, calculates their alignment index,
    and plots a clean comparison of parameter importance.
    """
    param_names = ["n", "f", "g", "h", "k", "L", "B*"]
    chunk = data_chunks[0]
    
    # --- 1. Setup Measurement (Doppler) Space ---
    active_pass_ids = chunk["pass_ids"][:num_passes]
    pass_mask = torch.isin(chunk["c"], active_pass_ids)
    
    t_active = chunk["t"][pass_mask]
    c_active = chunk["c"][pass_mask]
    t_mean_window = float(torch.mean(t_active))
    
    init_tle, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
    eval_ssv = state.MEE_SSV(init_tle=init_tle, num_measurements=len(t_active), fit_bstar=False)
    eval_ssv.add_linear_bias(name="pass_freq_bias", group_indices=c_active)
    x_eval = eval_ssv.get_initial_state()
    
    station_model = system.DifferentiableStation(
        lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, 
        ref_unix=t_mean_window, ref_gmst_rad=0.0 # simplified for jacobian
    )
    prop_eval = system.SGP4(ssv=eval_ssv)
    meas_eval = system.DopplerMeasurement(
        ssv=eval_ssv, station_model=station_model, freq_bias_group=eval_ssv.get_bias_group("pass_freq_bias"), time_bias_group=None
    )
    pipe_eval = system.MeasurementPipeline(propagator=prop_eval, measurement_model=meas_eval)
    t_since_meas = (t_active - t_mean_window) + 0.277
    
    # Compute Measurement Jacobian & PC1
    def meas_forward(x):
        return pipe_eval(x=x, tsince=t_since_meas, epoch=t_mean_window, center_freq=center_freq)
        
    J_meas = jacfwd(meas_forward)(x_eval)[:, :len(param_names)]
    J_meas_norm = J_meas / (torch.norm(J_meas, dim=0) + 1e-16)
    _, _, Vh_meas = torch.linalg.svd(J_meas_norm, full_matrices=False)
    v_meas = Vh_meas[0, :].detach().cpu().numpy() # PC1 of Measurement
    v_meas_importance = v_meas**2 # Fractional variance
    
    # --- 2. Setup Physical State (RIC) Space ---
    mask_24h = (t_gps >= epoch_unix) & (t_gps <= epoch_unix + 86400)
    t_eval_state = t_gps[mask_24h]
    r_eval_state = r_gps[mask_24h].to(torch.float64)
    v_eval_state = v_gps[mask_24h].to(torch.float64)
    t_since_state = (t_eval_state - epoch_unix) / 60.0
    
    def state_forward(x):
        r_calc, v_calc = prop_eval(x=x, tsince=t_since_state)
        # Simplified RIC magnitude mapping for variance
        R_hat = r_eval_state / torch.norm(r_eval_state, dim=1, keepdim=True)
        W_vec = torch.cross(r_eval_state, v_eval_state, dim=1)
        W_hat = W_vec / torch.norm(W_vec, dim=1, keepdim=True)
        S_hat = torch.cross(W_hat, R_hat, dim=1)
        
        pos_R = (r_calc * R_hat).sum(dim=1).unsqueeze(1)
        pos_I = (r_calc * S_hat).sum(dim=1).unsqueeze(1)
        pos_C = (r_calc * W_hat).sum(dim=1).unsqueeze(1)
        return torch.cat([pos_R, pos_I, pos_C], dim=1)

    J_state = jacfwd(state_forward)(x_eval).view(-1, len(x_eval))[:, :len(param_names)]
    J_state_norm = J_state / (torch.norm(J_state, dim=0) + 1e-16)
    _, _, Vh_state = torch.linalg.svd(J_state_norm, full_matrices=False)
    v_state = Vh_state[0, :].detach().cpu().numpy() # PC1 of Physical State
    v_state_importance = v_state**2
    
    # --- 3. Compute Alignment Index ---
    # Dot product of the principal eigenvectors
    alignment_index = np.abs(np.dot(v_meas, v_state))
    
    # --- 4. Plotting ---
    # Sort parameters by their importance to the Physical State
    sorted_indices = np.argsort(v_state_importance)[::-1]
    sorted_names = [param_names[i] for i in sorted_indices]
    sorted_state = v_state_importance[sorted_indices] * 100
    sorted_meas = v_meas_importance[sorted_indices] * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(sorted_names))
    width = 0.35
    
    ax.bar(x - width/2, sorted_state, width, label='Physical Error (RIC State PC1)', color='#1f77b4')
    ax.bar(x + width/2, sorted_meas, width, label='Sensor Sensitivity (Doppler PC1)', color='#ff7f0e')
    
    ax.set_title(f"Sensor-State Alignment Index: $\eta = {alignment_index:.3f}$\n({num_passes} Doppler Passes)")
    ax.set_ylabel('Fractional Parameter Importance (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# Execution:


import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import jacfwd
import dsgp4

def plot_full_observability_state_alignment(data_chunks, t_gps, r_gps, v_gps, tle_base, center_freq, epoch_unix, num_passes=3):
    """
    Computes the Generalized Observability-State Alignment Index by comparing 
    the full, variance-weighted parameter spaces of the sensor and the physical trajectory.
    """
    param_names = ["n", "f", "g", "h", "k", "L", "B*"]
    chunk = data_chunks[0]
    
    # --- 1. Setup Data & Base Evaluation ---
    active_pass_ids = chunk["pass_ids"][:num_passes]
    pass_mask = torch.isin(chunk["c"], active_pass_ids)
    
    t_active = chunk["t"][pass_mask]
    d_active_true = chunk["d_true"][pass_mask]
    c_active = chunk["c"][pass_mask]
    t_mean_window = float(torch.mean(t_active))
    
    init_tle, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
    eval_ssv = state.MEE_SSV(init_tle=init_tle, num_measurements=len(t_active), fit_bstar=False)
    eval_ssv.add_linear_bias(name="pass_freq_bias", group_indices=c_active)
    x_eval = eval_ssv.get_initial_state()
    
    station_model = system.DifferentiableStation(
        lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, 
        ref_unix=t_mean_window, ref_gmst_rad=0.0
    )
    prop_eval = system.SGP4(ssv=eval_ssv)
    meas_eval = system.DopplerMeasurement(
        ssv=eval_ssv, station_model=station_model, freq_bias_group=eval_ssv.get_bias_group("pass_freq_bias"), time_bias_group=None
    )
    pipe_eval = system.MeasurementPipeline(propagator=prop_eval, measurement_model=meas_eval)
    t_since_meas = (t_active - t_mean_window) + 0.277
    
    # --- 2. Measurement Space (Doppler) PCA & Weighting ---
    with torch.no_grad():
        d_pred_init = pipe_eval(x=x_eval, tsince=t_since_meas, epoch=t_mean_window, center_freq=center_freq)
    residuals_meas = d_active_true - d_pred_init
    
    def meas_forward(x):
        return pipe_eval(x=x, tsince=t_since_meas, epoch=t_mean_window, center_freq=center_freq)
        
    J_meas = jacfwd(meas_forward)(x_eval)[:, :len(param_names)]
    J_meas_norm = J_meas / (torch.norm(J_meas, dim=0) + 1e-16)
    U_m, _, Vh_m = torch.linalg.svd(J_meas_norm, full_matrices=False)
    V_m = Vh_m.T
    
    # Variance explained by each measurement PC
    c_m = U_m.T @ residuals_meas
    w_m = (c_m**2).detach().cpu().numpy()
    w_m = w_m / (np.sum(w_m) + 1e-16)  # Normalize weights to sum to 1
    
    # Measurement Attribution Matrix
    C_m = V_m.detach().cpu().numpy() @ np.diag(w_m) @ Vh_m.detach().cpu().numpy()
    
    # --- 3. Physical State (RIC) PCA & Weighting ---
    mask_24h = (t_gps >= epoch_unix) & (t_gps <= epoch_unix + 86400)
    t_eval_state = t_gps[mask_24h]
    r_eval_state = r_gps[mask_24h].to(torch.float64)
    v_eval_state = v_gps[mask_24h].to(torch.float64)
    t_since_state = (t_eval_state - epoch_unix) / 60.0
    
    def state_forward(x):
        r_calc, v_calc = prop_eval(x=x, tsince=t_since_state)
        R_hat = r_eval_state / torch.norm(r_eval_state, dim=1, keepdim=True)
        W_vec = torch.cross(r_eval_state, v_eval_state, dim=1)
        W_hat = W_vec / torch.norm(W_vec, dim=1, keepdim=True)
        S_hat = torch.cross(W_hat, R_hat, dim=1)
        
        pos_R = (r_calc * R_hat).sum(dim=1).unsqueeze(1)
        pos_I = (r_calc * S_hat).sum(dim=1).unsqueeze(1)
        pos_C = (r_calc * W_hat).sum(dim=1).unsqueeze(1)
        return torch.cat([pos_R, pos_I, pos_C], dim=1)

    with torch.no_grad():
        r_prior = state_forward(x_eval)
        residuals_state = r_prior.view(-1)
        
    J_state = jacfwd(state_forward)(x_eval).view(-1, len(x_eval))[:, :len(param_names)]
    J_state_norm = J_state / (torch.norm(J_state, dim=0) + 1e-16)
    U_s, _, Vh_s = torch.linalg.svd(J_state_norm, full_matrices=False)
    V_s = Vh_s.T
    
    # Variance explained by each physical state PC
    c_s = U_s.T @ residuals_state
    w_s = (c_s**2).detach().cpu().numpy()
    w_s = w_s / (np.sum(w_s) + 1e-16) # Normalize weights to sum to 1
    
    # State Attribution Matrix
    C_s = V_s.detach().cpu().numpy() @ np.diag(w_s) @ Vh_s.detach().cpu().numpy()
    
    # --- 4. Compute Generalized Alignment Index (RV Coefficient) ---
    # This is exactly your proposed weighted sum of vector correlations, normalized to [0, 1]
    inner_product = np.trace(C_m @ C_s)
    norm_m = np.linalg.norm(C_m, ord='fro')
    norm_s = np.linalg.norm(C_s, ord='fro')
    alignment_index = inner_product / (norm_m * norm_s + 1e-16)
    
    # --- 5. Extract Total Parameter Importances for Plotting ---
    # The diagonal of the attribution matrices gives the total weighted parameter importance
    param_importance_m = np.diag(C_m) * 100
    param_importance_s = np.diag(C_s) * 100
    
    # --- 6. Plotting ---
    sorted_indices = np.argsort(param_importance_s)[::-1]
    sorted_names = [param_names[i] for i in sorted_indices]
    sorted_state = param_importance_s[sorted_indices]
    sorted_meas = param_importance_m[sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sorted_names))
    width = 0.35
    
    ax.bar(x - width/2, sorted_state, width, label='Physical State Attribution ($C_{\mathcal{X}}$)', color='#1f77b4')
    ax.bar(x + width/2, sorted_meas, width, label='Sensor Measurement Attribution ($C_{\mathcal{H}}$)', color='#ff7f0e')
    
    ax.set_title(f"Generalized Observability-State Alignment: $\eta_{{total}} = {alignment_index:.3f}$\n({num_passes} Doppler Passes)")
    ax.set_ylabel('Total Parameter Importance (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import jacfwd
import dsgp4

def plot_decoupled_state_alignment(data_chunks, t_gps, r_gps, v_gps, tle_base, center_freq, epoch_unix, num_passes=3):
    """
    Computes parameter attribution separately for the Doppler Sensor, 
    Physical Position (RIC), and Physical Velocity (RIC).
    """
    param_names = ["n", "f", "g", "h", "k", "L"]
    chunk = data_chunks[0]
    
    # --- 1. Setup Data & Base Evaluation ---
    active_pass_ids = chunk["pass_ids"][:num_passes]
    pass_mask = torch.isin(chunk["c"], active_pass_ids)
    
    t_active = chunk["t"][pass_mask]
    d_active_true = chunk["d_true"][pass_mask]
    t_mean_window = float(torch.mean(t_active))
    
    init_tle, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
    eval_ssv = state.MEE_SSV(init_tle=init_tle, num_measurements=len(t_active), fit_bstar=False)
    eval_ssv.add_linear_bias(name="pass_freq_bias", group_indices=chunk["c"][pass_mask])
    x_eval = eval_ssv.get_initial_state()
    
    station_model = system.DifferentiableStation(
        lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, ref_unix=t_mean_window, ref_gmst_rad=0.0
    )
    prop_eval = system.SGP4(ssv=eval_ssv)
    meas_eval = system.DopplerMeasurement(
        ssv=eval_ssv, station_model=station_model, freq_bias_group=eval_ssv.get_bias_group("pass_freq_bias"), time_bias_group=None
    )
    pipe_eval = system.MeasurementPipeline(propagator=prop_eval, measurement_model=meas_eval)
    t_since_meas = (t_active - t_mean_window) + 0.277
    
    # --- 2. Measurement Space (Doppler) Attribution ---
    with torch.no_grad():
        d_pred_init = pipe_eval(x=x_eval, tsince=t_since_meas, epoch=t_mean_window, center_freq=center_freq)
    residuals_meas = d_active_true - d_pred_init
    
    def meas_forward(x):
        return pipe_eval(x=x, tsince=t_since_meas, epoch=t_mean_window, center_freq=center_freq)
        
    J_meas = jacfwd(meas_forward)(x_eval)[:, :len(param_names)]
    J_meas_norm = J_meas / (torch.norm(J_meas, dim=0) + 1e-16)
    U_m, _, Vh_m = torch.linalg.svd(J_meas_norm, full_matrices=False)
    
    w_m = (U_m.T @ residuals_meas)**2
    w_m = (w_m / (torch.sum(w_m) + 1e-16)).detach().cpu().numpy()
    C_m = Vh_m.T.detach().cpu().numpy() @ np.diag(w_m) @ Vh_m.detach().cpu().numpy()
    
    # --- 3. Decoupled Physical State (Position vs Velocity) ---
    mask_24h = (t_gps >= epoch_unix) & (t_gps <= epoch_unix + 86400)
    t_eval_state = t_gps[mask_24h]
    r_eval_state = r_gps[mask_24h].to(torch.float64)
    v_eval_state = v_gps[mask_24h].to(torch.float64)
    t_since_state = (t_eval_state - epoch_unix) / 60.0
    
    def state_forward_decoupled(x):
        r_calc, v_calc = prop_eval(x=x, tsince=t_since_state)
        R_hat = r_eval_state / torch.norm(r_eval_state, dim=1, keepdim=True)
        W_vec = torch.cross(r_eval_state, v_eval_state, dim=1)
        W_hat = W_vec / torch.norm(W_vec, dim=1, keepdim=True)
        S_hat = torch.cross(W_hat, R_hat, dim=1)
        
        # Position RIC
        pos_R = (r_calc * R_hat).sum(dim=1).unsqueeze(1)
        pos_I = (r_calc * S_hat).sum(dim=1).unsqueeze(1)
        pos_C = (r_calc * W_hat).sum(dim=1).unsqueeze(1)
        pos_ric = torch.cat([pos_R, pos_I, pos_C], dim=1)
        
        # Velocity RIC
        vel_R = (v_calc * R_hat).sum(dim=1).unsqueeze(1)
        vel_I = (v_calc * S_hat).sum(dim=1).unsqueeze(1)
        vel_C = (v_calc * W_hat).sum(dim=1).unsqueeze(1)
        vel_ric = torch.cat([vel_R, vel_I, vel_C], dim=1)
        
        return pos_ric, vel_ric

    with torch.no_grad():
        r_prior, v_prior = state_forward_decoupled(x_eval)
        res_pos = r_prior.view(-1)
        res_vel = v_prior.view(-1)
        
    # Get Jacobians
    J_full = jacfwd(state_forward_decoupled)(x_eval)
    J_pos = J_full[0].view(-1, len(x_eval))[:, :len(param_names)]
    J_vel = J_full[1].view(-1, len(x_eval))[:, :len(param_names)]
    
    # Position SVD
    J_pos_norm = J_pos / (torch.norm(J_pos, dim=0) + 1e-16)
    U_p, _, Vh_p = torch.linalg.svd(J_pos_norm, full_matrices=False)
    w_p = (U_p.T @ res_pos)**2
    w_p = (w_p / (torch.sum(w_p) + 1e-16)).detach().cpu().numpy()
    C_p = Vh_p.T.detach().cpu().numpy() @ np.diag(w_p) @ Vh_p.detach().cpu().numpy()
    
    # Velocity SVD
    J_vel_norm = J_vel / (torch.norm(J_vel, dim=0) + 1e-16)
    U_v, _, Vh_v = torch.linalg.svd(J_vel_norm, full_matrices=False)
    w_v = (U_v.T @ res_vel)**2
    w_v = (w_v / (torch.sum(w_v) + 1e-16)).detach().cpu().numpy()
    C_v = Vh_v.T.detach().cpu().numpy() @ np.diag(w_v) @ Vh_v.detach().cpu().numpy()
    
    # --- 4. Plotting ---
    imp_meas = np.diag(C_m) * 100
    imp_pos = np.diag(C_p) * 100
    imp_vel = np.diag(C_v) * 100
    
    sorted_indices = np.argsort(imp_pos)[::-1]
    sorted_names = [param_names[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(sorted_names))
    width = 0.25
    
    ax.bar(x - width, imp_pos[sorted_indices], width, label='Position Error Attribution', color='#1f77b4')
    ax.bar(x, imp_vel[sorted_indices], width, label='Velocity Error Attribution', color='#2ca02c')
    ax.bar(x + width, imp_meas[sorted_indices], width, label='Doppler Sensor Attribution', color='#ff7f0e')
    
    ax.set_title(f"Decoupled Kinematic Attribution Analysis ({num_passes} Passes)")
    ax.set_ylabel('Total Parameter Importance (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# Execution:

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.func import jacfwd
import dsgp4

def plot_pas_and_correlation(data_chunks, t_gps, r_gps, v_gps, tle_base, center_freq, epoch_unix, num_passes=3):
    """
    Plots the Parameter Alignment Score (PAS) alongside the Parameter Correlation Matrix
    to justify parameter selection based on high utility and low collinearity.
    """
    param_names = ["n", "f", "g", "h", "k", "L"]
    chunk = data_chunks[0]
    
    # --- 1. Setup Data & Base Evaluation ---
    active_pass_ids = chunk["pass_ids"][:num_passes]
    pass_mask = torch.isin(chunk["c"], active_pass_ids)
    
    t_active = chunk["t"][pass_mask]
    d_active_true = chunk["d_true"][pass_mask]
    c_active = chunk["c"][pass_mask]
    t_mean_window = float(torch.mean(t_active))
    
    init_tle, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
    eval_ssv = state.MEE_SSV(init_tle=init_tle, num_measurements=len(t_active), fit_bstar=False)
    eval_ssv.add_linear_bias(name="pass_freq_bias", group_indices=c_active)
    x_eval = eval_ssv.get_initial_state()
    
    station_model = system.DifferentiableStation(
        lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, ref_unix=t_mean_window, ref_gmst_rad=0.0
    )
    prop_eval = system.SGP4(ssv=eval_ssv)
    meas_eval = system.DopplerMeasurement(
        ssv=eval_ssv, station_model=station_model, freq_bias_group=eval_ssv.get_bias_group("pass_freq_bias"), time_bias_group=None
    )
    pipe_eval = system.MeasurementPipeline(propagator=prop_eval, measurement_model=meas_eval)
    t_since_meas = (t_active - t_mean_window) + 0.277
    
    # --- 2. Measurement Space: Attribution & Correlation ---
    with torch.no_grad():
        d_pred_init = pipe_eval(x=x_eval, tsince=t_since_meas, epoch=t_mean_window, center_freq=center_freq)
    residuals_meas = d_active_true - d_pred_init
    
    def meas_forward(x):
        return pipe_eval(x=x, tsince=t_since_meas, epoch=t_mean_window, center_freq=center_freq)
        
    J_meas = jacfwd(meas_forward)(x_eval)[:, :len(param_names)]
    
    # A. Measurement Attribution via SVD
    J_meas_norm = J_meas / (torch.norm(J_meas, dim=0) + 1e-16)
    U_m, _, Vh_m = torch.linalg.svd(J_meas_norm, full_matrices=False)
    w_m = (U_m.T @ residuals_meas)**2
    w_m = (w_m / (torch.sum(w_m) + 1e-16)).detach().cpu().numpy()
    C_m = Vh_m.T.detach().cpu().numpy() @ np.diag(w_m) @ Vh_m.detach().cpu().numpy()
    imp_meas = np.diag(C_m)
    
    # B. Parameter Correlation via Fisher Information Matrix
    sigma_obs = 20.0
    W = 1.0 / (sigma_obs**2)
    FIM = J_meas.T @ (J_meas * W)
    Cov = torch.linalg.pinv(FIM)
    std_dev = torch.sqrt(torch.abs(torch.diag(Cov)))
    outer_std = torch.outer(std_dev, std_dev) + 1e-16
    Corr_np = (Cov / outer_std).detach().cpu().numpy()
    
    # --- 3. Physical State Space: Decoupled Attribution ---
    mask_24h = (t_gps >= epoch_unix) & (t_gps <= epoch_unix + 86400)
    t_eval_state = t_gps[mask_24h]
    r_eval_state = r_gps[mask_24h].to(torch.float64)
    v_eval_state = v_gps[mask_24h].to(torch.float64)
    t_since_state = (t_eval_state - epoch_unix) / 60.0
    
    def state_forward_decoupled(x):
        r_calc, v_calc = prop_eval(x=x, tsince=t_since_state)
        R_hat = r_eval_state / torch.norm(r_eval_state, dim=1, keepdim=True)
        W_vec = torch.cross(r_eval_state, v_eval_state, dim=1)
        W_hat = W_vec / torch.norm(W_vec, dim=1, keepdim=True)
        S_hat = torch.cross(W_hat, R_hat, dim=1)
        
        pos_ric = torch.cat([(r_calc * R_hat).sum(dim=1).unsqueeze(1), (r_calc * S_hat).sum(dim=1).unsqueeze(1), (r_calc * W_hat).sum(dim=1).unsqueeze(1)], dim=1)
        vel_ric = torch.cat([(v_calc * R_hat).sum(dim=1).unsqueeze(1), (v_calc * S_hat).sum(dim=1).unsqueeze(1), (v_calc * W_hat).sum(dim=1).unsqueeze(1)], dim=1)
        return pos_ric, vel_ric

    with torch.no_grad():
        r_prior, v_prior = state_forward_decoupled(x_eval)
        res_pos, res_vel = r_prior.view(-1), v_prior.view(-1)
        
    J_full = jacfwd(state_forward_decoupled)(x_eval)
    J_pos = J_full[0].view(-1, len(x_eval))[:, :len(param_names)]
    J_vel = J_full[1].view(-1, len(x_eval))[:, :len(param_names)]
    
    # Position SVD
    J_pos_norm = J_pos / (torch.norm(J_pos, dim=0) + 1e-16)
    U_p, _, Vh_p = torch.linalg.svd(J_pos_norm, full_matrices=False)
    w_p = (U_p.T @ res_pos)**2
    w_p = (w_p / (torch.sum(w_p) + 1e-16)).detach().cpu().numpy()
    imp_pos = np.diag(Vh_p.T.detach().cpu().numpy() @ np.diag(w_p) @ Vh_p.detach().cpu().numpy())
    
    # Velocity SVD
    J_vel_norm = J_vel / (torch.norm(J_vel, dim=0) + 1e-16)
    U_v, _, Vh_v = torch.linalg.svd(J_vel_norm, full_matrices=False)
    w_v = (U_v.T @ res_vel)**2
    w_v = (w_v / (torch.sum(w_v) + 1e-16)).detach().cpu().numpy()
    imp_vel = np.diag(Vh_v.T.detach().cpu().numpy() @ np.diag(w_v) @ Vh_v.detach().cpu().numpy())
    
    # --- 4. Calculate Parameter Alignment Scores (PAS) ---
    # Geometric mean of measurement attribution and physical attribution
    pas_pos = np.sqrt(imp_meas * imp_pos) * 100
    pas_vel = np.sqrt(imp_meas * imp_vel) * 100
    
    # --- 5. Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot A: PAS Scores
    sorted_indices = np.argsort(pas_pos)[::-1]
    sorted_names = [param_names[i] for i in sorted_indices]
    
    x = np.arange(len(sorted_names))
    width = 0.35
    
    axes[0].bar(x - width/2, pas_pos[sorted_indices], width, label='PAS (Position Error)', color='#1f77b4')
    axes[0].bar(x + width/2, pas_vel[sorted_indices], width, label='PAS (Velocity Error)', color='#2ca02c')
    
    axes[0].set_title(f"Parameter Alignment Score (PAS)\n({num_passes} Passes)")
    axes[0].set_ylabel('Alignment Index (%)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sorted_names)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Plot B: Parameter Correlation Heatmap
    sns.heatmap(
        Corr_np, annot=True, cmap="coolwarm", vmin=-1, vmax=1, 
        xticklabels=param_names, yticklabels=param_names, fmt=".2f",
        ax=axes[1], square=True, cbar_kws={"shrink": .8}
    )
    axes[1].set_title(f"Parameter Correlation Matrix\n({num_passes} Passes)")
    axes[1].set_xlabel("Modified Equinoctial Elements")
    axes[1].set_ylabel("Modified Equinoctial Elements")

    plt.tight_layout()
    plt.show()

# Execution:
# plot_pas_and_correlation(data_chunks, t_gps, r_gps, v_gps, tle_base, center_freq, epoch_unix, num_passes=3)

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import jacfwd
import dsgp4
import diffod.state as state
import diffod.functional.system as system
from diffod.utils import unix_to_mjd

def plot_residual_explainability_r2(data_chunks, t_gps, r_gps, v_gps, tle_base, center_freq, epoch_unix, num_passes=3):
    """
    Computes the Single-Parameter Explained Variance (R^2) by calculating the 
    squared cosine similarity between the Jacobian columns and the true residuals.
    """
    param_names = ["n", "f", "g", "h", "k", "L", "B*"]
    chunk = data_chunks[0]
    
    # --- 1. Setup Data ---
    active_pass_ids = chunk["pass_ids"][:num_passes]
    pass_mask = torch.isin(chunk["c"], active_pass_ids)
    
    t_active = chunk["t"][pass_mask]
    d_active_true = chunk["d_true"][pass_mask]
    t_mean_window = float(torch.mean(t_active))
    
    init_tle, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
    eval_ssv = state.MEE_SSV(init_tle=init_tle, num_measurements=len(t_active), fit_bstar=False)
    eval_ssv.add_linear_bias(name="pass_freq_bias", group_indices=chunk["c"][pass_mask])
    x_eval = eval_ssv.get_initial_state()
    
    station_model = system.DifferentiableStation(
        lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, ref_unix=t_mean_window, ref_gmst_rad=0.0
    )
    prop_eval = system.SGP4(ssv=eval_ssv)
    meas_eval = system.DopplerMeasurement(
        ssv=eval_ssv, station_model=station_model, freq_bias_group=eval_ssv.get_bias_group("pass_freq_bias"), time_bias_group=None
    )
    pipe_eval = system.MeasurementPipeline(propagator=prop_eval, measurement_model=meas_eval)
    t_since_meas = (t_active - t_mean_window) + 0.277
    
    # --- 2. Compute Doppler R^2 ---
    def meas_forward(x):
        return pipe_eval(x=x, tsince=t_since_meas, epoch=t_mean_window, center_freq=center_freq)
        
    with torch.no_grad():
        d_pred = meas_forward(x_eval)
        r_dopp = (d_active_true - d_pred).view(-1)
        
    J_dopp = jacfwd(meas_forward)(x_eval)[:, :len(param_names)]
    
    r2_dopp = []
    for i in range(len(param_names)):
        J_i = J_dopp[:, i]
        cos_sim = torch.dot(J_i, r_dopp) / (torch.norm(J_i) * torch.norm(r_dopp) + 1e-16)
        r2_dopp.append((cos_sim**2).item() * 100)
        
    # --- 3. Compute Physical State (Position & Velocity) R^2 ---
    mask_24h = (t_gps >= epoch_unix) & (t_gps <= epoch_unix + 86400)
    t_eval_state = t_gps[mask_24h]
    r_eval_state = r_gps[mask_24h].to(torch.float64)
    v_eval_state = v_gps[mask_24h].to(torch.float64)
    t_since_state = (t_eval_state - epoch_unix) / 60.0
    
    def state_forward_decoupled(x):
        r_calc, v_calc = prop_eval(x=x, tsince=t_since_state)
        # We compute raw Cartesian residuals for variance to avoid RIC projection artifacts
        return r_calc, v_calc

    with torch.no_grad():
        r_calc, v_calc = state_forward_decoupled(x_eval)
        r_pos = (r_eval_state - r_calc).view(-1)
        r_vel = (v_eval_state - v_calc).view(-1)
        
    J_full = jacfwd(state_forward_decoupled)(x_eval)
    J_pos = J_full[0].view(-1, len(x_eval))[:, :len(param_names)]
    J_vel = J_full[1].view(-1, len(x_eval))[:, :len(param_names)]
    
    r2_pos = []
    r2_vel = []
    for i in range(len(param_names)):
        J_pi = J_pos[:, i]
        cos_sim_p = torch.dot(J_pi, r_pos) / (torch.norm(J_pi) * torch.norm(r_pos) + 1e-16)
        r2_pos.append((cos_sim_p**2).item() * 100)
        
        J_vi = J_vel[:, i]
        cos_sim_v = torch.dot(J_vi, r_vel) / (torch.norm(J_vi) * torch.norm(r_vel) + 1e-16)
        r2_vel.append((cos_sim_v**2).item() * 100)
        
    # --- 4. Plotting ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(param_names))
    width = 0.25
    
    ax.bar(x - width, r2_pos, width, label='Position Error Explained', color='#1f77b4')
    ax.bar(x, r2_vel, width, label='Velocity Error Explained', color='#2ca02c')
    ax.bar(x + width, r2_dopp, width, label='Doppler Error Explained', color='#ff7f0e')
    
    ax.set_title(f"Single-Parameter Explained Variance ($R^2$)\n({num_passes} Passes, 24h Forecast)")
    ax.set_ylabel('Variance Explained by Parameter (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# Execution:
# plot_residual_explainability_r2(data_chunks, t_gps, r_gps, v_gps, tle_base, center_freq, epoch_unix, num_passes=3)

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import hessian
import dsgp4

def plot_exact_hessian_alignment(data_chunks, t_gps, r_gps, tle_base, center_freq, epoch_unix, num_passes=3):
    """
    Computes the Exact Hessian of the Sensor Loss and Physical Trajectory Loss,
    evaluates their alignment, and plots the raw curvature per parameter.
    """
    param_names = ["n", "f", "g", "h", "k", "L", "B*"]
    chunk = data_chunks[0]
    
    # --- Setup Base Evaluation ---
    active_pass_ids = chunk["pass_ids"][:num_passes]
    pass_mask = torch.isin(chunk["c"], active_pass_ids)
    
    t_active = chunk["t"][pass_mask]
    d_active_true = chunk["d_true"][pass_mask]
    t_mean_window = float(torch.mean(t_active))
    
    init_tle, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
    eval_ssv = state.MEE_SSV(init_tle=init_tle, num_measurements=len(t_active), fit_bstar=False)
    eval_ssv.add_linear_bias(name="pass_freq_bias", group_indices=chunk["c"][pass_mask])
    x_eval = eval_ssv.get_initial_state()
    
    station_model = system.DifferentiableStation(
        lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, ref_unix=t_mean_window, ref_gmst_rad=0.0
    )
    prop_eval = system.SGP4(ssv=eval_ssv)
    meas_eval = system.DopplerMeasurement(
        ssv=eval_ssv, station_model=station_model, freq_bias_group=eval_ssv.get_bias_group("pass_freq_bias"), time_bias_group=None
    )
    pipe_eval = system.MeasurementPipeline(propagator=prop_eval, measurement_model=meas_eval)
    t_since_meas = (t_active - t_mean_window) + 0.277
    
    mask_24h = (t_gps >= epoch_unix) & (t_gps <= epoch_unix + 86400)
    t_eval_state = t_gps[mask_24h]
    r_eval_state = r_gps[mask_24h].to(torch.float64)
    t_since_state = (t_eval_state - epoch_unix) / 60.0

    # --- 1. Define Scalar Objective Functions (Loss) ---
    def sensor_loss_fn(x):
        # Mean Squared Error of the Doppler curve
        d_pred = pipe_eval(x=x, tsince=t_since_meas, epoch=t_mean_window, center_freq=center_freq)
        return torch.mean((d_active_true - d_pred)**2)

    def trajectory_loss_fn(x):
        # Mean Squared Error of the 3D Position over 24 hours
        r_calc, _ = prop_eval(x=x, tsince=t_since_state)
        return torch.mean(torch.sum((r_eval_state - r_calc)**2, dim=1))

    # --- 2. Compute Exact Hessians ---
    # hessian() computes the full matrix of second partial derivatives
    H_meas_full = hessian(sensor_loss_fn)(x_eval)
    H_state_full = hessian(trajectory_loss_fn)(x_eval)
    
    # Isolate the orbital parameters (top-left 7x7 block)
    H_meas = H_meas_full[:len(param_names), :len(param_names)]
    H_state = H_state_full[:len(param_names), :len(param_names)]
    
    # Normalize Hessians to compare their structural geometry, not raw magnitudes
    H_meas_norm = H_meas / (torch.norm(H_meas, p='fro') + 1e-16)
    H_state_norm = H_state / (torch.norm(H_state, p='fro') + 1e-16)
    
    # --- 3. Compute Hessian Alignment Metric ---
    # Frobenius inner product = trace(A^T B)
    hessian_alignment = torch.trace(H_meas_norm.T @ H_state_norm).item()
    
    # --- 4. Extract Curvature for Plotting ---
    # The absolute diagonal elements represent the pure convexity/steepness per parameter
    curv_meas = torch.abs(torch.diag(H_meas_norm)).detach().cpu().numpy()
    curv_state = torch.abs(torch.diag(H_state_norm)).detach().cpu().numpy()
    
    # --- 5. Plotting ---
    sorted_indices = np.argsort(curv_state)[::-1]
    sorted_names = [param_names[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sorted_names))
    width = 0.35
    
    ax.bar(x - width/2, curv_state[sorted_indices] * 100, width, label='Physical Trajectory Curvature ($H_{state}$)', color='#1f77b4')
    ax.bar(x + width/2, curv_meas[sorted_indices] * 100, width, label='Sensor Curvature ($H_{meas}$)', color='#ff7f0e')
    
    ax.set_title(f"Exact Hessian Curvature Alignment: $\eta_{{Hessian}} = {hessian_alignment:.3f}$\n({num_passes} Doppler Passes)")
    ax.set_ylabel('Relative Loss Landscape Curvature (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# Execution:
# plot_exact_hessian_alignment(data_chunks, t_gps, r_gps, tle_base, center_freq, epoch_unix, num_passes=3)

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import grad, hessian
import dsgp4
import diffod.state as state
import diffod.functional.system as system
from diffod.utils import unix_to_mjd
from astropy.time import Time

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from torch.autograd.functional import grad, hessian

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import grad, hessian
import dsgp4
import diffod.state as state
import diffod.functional.system as system
from diffod.utils import unix_to_mjd
from astropy.time import Time

def plot_objective_sensitivity_alignment(data_chunks, t_gps, r_gps, v_gps, tle_base, center_freq, epoch_unix, num_passes=3):
    """
    Computes the Gradient (1st-order) and Hessian diagonal (2nd-order) of the 
    RMSE objective functions for Doppler, Position, and Velocity.
    """
    param_names = ["n", "f", "g", "h", "k", "L", "B*"]
    num_params = len(param_names)
    chunk = data_chunks[0]
    
    # --- 1. Setup Data & Forward Models ---
    active_pass_ids = chunk["pass_ids"][:num_passes]
    pass_mask = torch.isin(chunk["c"], active_pass_ids)
    
    t_active = chunk["t"][pass_mask]
    d_active_true = chunk["d_true"][pass_mask]
    c_active = chunk["c"][pass_mask]
    t_mean_window = float(torch.mean(t_active))
    
    init_tle, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
    eval_ssv = state.MEE_SSV(init_tle=init_tle, num_measurements=len(t_active), fit_bstar=False)
    eval_ssv.add_linear_bias(name="pass_freq_bias", group_indices=c_active)
    x_eval = eval_ssv.get_initial_state()
    
    station_model = system.DifferentiableStation(
        lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, ref_unix=t_mean_window, ref_gmst_rad=0.0
    )
    prop_eval = system.SGP4(ssv=eval_ssv, use_pretrained_model=False)
    meas_eval = system.DopplerMeasurement(
        ssv=eval_ssv, station_model=station_model, freq_bias_group=eval_ssv.get_bias_group("pass_freq_bias"), time_bias_group=None
    )
    pipe_eval = system.MeasurementPipeline(propagator=prop_eval, measurement_model=meas_eval)
    
    t_since_meas = (t_active - t_mean_window) + 0.277
    
    mask_24h = (t_gps >= epoch_unix) & (t_gps <= epoch_unix + 86400)
    t_eval_state = t_gps[mask_24h]
    r_eval_state = r_gps[mask_24h].to(torch.float64)
    v_eval_state = v_gps[mask_24h].to(torch.float64)
    t_since_state = (t_eval_state - epoch_unix) / 60.0
    
    # --- 2. Define Scalar Objective Functions (RMSE) ---
    def loss_doppler(x):
        d_pred = pipe_eval(x=x, tsince=t_since_meas, epoch=t_mean_window, center_freq=center_freq)
        return torch.sqrt(torch.mean((d_active_true - d_pred)**2))
        
    def loss_pos(x):
        r_calc, _ = prop_eval(x=x, tsince=t_since_state)
        # RMSE of 3D distance
        return torch.sqrt(torch.mean(torch.sum((r_calc - r_eval_state)**2, dim=1)))

    def loss_vel(x):
        _, v_calc = prop_eval(x=x, tsince=t_since_state)
        return torch.sqrt(torch.mean(torch.sum((v_calc - v_eval_state)**2, dim=1)))

    # --- 3. Compute Gradients and Hessians ---
    def extract_sensitivities(loss_fn):
        # 1st Order: Gradient vector
        g = grad(loss_fn)(x_eval)[:num_params]
        # 2nd Order: Hessian matrix -> Extract absolute diagonal (curvature)
        h = hessian(loss_fn)(x_eval)[:num_params, :num_params]
        h_diag = torch.abs(torch.diag(h))
        
        # Normalize to strictly compare directional shapes (L2 norm)
        g_norm = (torch.abs(g) / torch.norm(g)).detach().cpu().numpy()
        h_norm = (h_diag / torch.norm(h_diag)).detach().cpu().numpy()
        return np.abs(g.detach().cpu().numpy()), np.abs(h_diag.detach().cpu().numpy())

    g_dopp, h_dopp = extract_sensitivities(loss_doppler)
    g_pos, h_pos = extract_sensitivities(loss_pos)
    g_vel, h_vel = extract_sensitivities(loss_vel)
    
    # --- 4. Compute Alignment Metrics (Cosine Similarities) ---
    # Gradient Alignment (How well does the sensor error direction match the physical error direction?)
    align_grad_pos = np.dot(g_dopp, g_pos)
    align_grad_vel = np.dot(g_dopp, g_vel)
    
    # Hessian Alignment (How well does the sensor curvature match the physical curvature?)
    align_hess_pos = np.dot(h_dopp, h_pos)
    align_hess_vel = np.dot(h_dopp, h_vel)
    
    # --- 5. Plotting ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    x = np.arange(num_params)
    width = 0.25
    
    # Plot 1st-Order (Gradients)
    axes[0].bar(x - width, np.log(g_pos), width, label='Position Gradient', color='#1f77b4')
    axes[0].bar(x, np.log(g_vel), width, label='Velocity Gradient', color='#2ca02c')
    axes[0].bar(x + width, np.log(g_dopp), width, label='Doppler Sensor Gradient', color='#ff7f0e')
    axes[0].set_title(f"1st-Order Sensitivity (RMSE Gradients)")
    axes[0].set_ylabel("Logarithmic Gradient Magnitude")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(param_names)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Plot 2nd-Order (Hessian Curvature)
    axes[1].bar(x - width, np.log(h_pos), width, label='Position Curvature', color='#1f77b4')
    axes[1].bar(x, np.log(h_vel), width, label='Velocity Curvature', color='#2ca02c')
    axes[1].bar(x + width, np.log(h_dopp), width, label='Doppler Sensor Curvature', color='#ff7f0e')
    axes[1].set_title(f"2nd-Order Sensitivity (RMSE Hessian Diagonal)")
    axes[1].set_ylabel("Logarithmic Curvature Magnitude")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(param_names)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# Execution:
# plot_objective_sensitivity_alignment(data_chunks, t_gps, r_gps, v_gps, tle_base, center_freq, epoch_unix, num_passes=3)
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import grad, hessian
import dsgp4
import diffod.state as state
import diffod.functional.system as system
from diffod.utils import unix_to_mjd
from astropy.time import Time

def analyze_hessian_topology(data_chunks, t_gps, r_gps, v_gps, tle_base, center_freq, epoch_unix, num_passes=3):
    """
    Computes the parameter-scaled Hessian and analyzes its eigenspectrum 
    to detect flat valleys and saddle points in the loss landscape.
    """
    param_names = ["n", "f", "g", "h", "k", "L", "B*"]
    num_params = len(param_names)
    chunk = data_chunks[0]
    
    # --- 1. Setup ---
    active_pass_ids = chunk["pass_ids"][:num_passes]
    pass_mask = torch.isin(chunk["c"], active_pass_ids)
    
    t_active = chunk["t"][pass_mask]
    d_active_true = chunk["d_true"][pass_mask]
    t_mean_window = float(torch.mean(t_active))
    
    init_tle, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
    eval_ssv = state.MEE_SSV(init_tle=init_tle, num_measurements=len(t_active), fit_bstar=False)
    eval_ssv.add_linear_bias(name="pass_freq_bias", group_indices=chunk["c"][pass_mask])
    x_eval = eval_ssv.get_initial_state()
    
    # Scale matrix: Diagonal matrix of the absolute parameter values
    # We add a small epsilon to avoid scaling by zero for highly circular/equatorial orbits
    x_scale = torch.abs(x_eval[:num_params]) + 1e-6
    S = torch.diag(x_scale)
    
    # --- 2. Setup Forward Models ---
    station_model = system.DifferentiableStation(
        lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, ref_unix=t_mean_window, ref_gmst_rad=0.0
    )
    prop_eval = system.SGP4(ssv=eval_ssv)
    meas_eval = system.DopplerMeasurement(
        ssv=eval_ssv, station_model=station_model, freq_bias_group=eval_ssv.get_bias_group("pass_freq_bias"), time_bias_group=None
    )
    pipe_eval = system.MeasurementPipeline(propagator=prop_eval, measurement_model=meas_eval)
    
    t_since_meas = (t_active - t_mean_window) + 0.277
    
    def loss_doppler(x):
        d_pred = pipe_eval(x=x, tsince=t_since_meas, epoch=t_mean_window, center_freq=center_freq)
        return torch.sqrt(torch.mean((d_active_true - d_pred)**2))

    # --- 3. Compute Full Hessian and Apply Scaling ---
    # Extract the raw 7x7 Hessian
    H_raw = hessian(loss_doppler)(x_eval)[:num_params, :num_params]
    
    # Scale the Hessian: H_scaled = S * H_raw * S
    # This transforms the Hessian into "fractional variance" space, making parameters comparable
    H_scaled = S @ H_raw @ S
    
    # --- 4. Eigensystem Analysis (Saddle Point Detection) ---
    evals, evecs = torch.linalg.eigh(H_scaled)
    evals_np = evals.detach().cpu().numpy()
    evecs_np = evecs.detach().cpu().numpy()
    
    # Detect negative eigenvalues
    is_saddle = np.any(evals_np < -1e-8) # Small threshold for numerical noise
    
    # --- 5. Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot A: The Eigenspectrum
    colors = ['red' if val < 0 else 'blue' for val in evals_np]
    axes[0].bar(range(1, num_params + 1), evals_np, color=colors, alpha=0.8)
    axes[0].axhline(0, color='black', linewidth=1)
    
    title_suffix = "SADDLE POINT DETECTED" if is_saddle else "Convex (Local Minimum)"
    axes[0].set_title(f"Scaled Hessian Eigenspectrum\nTopology: {title_suffix}")
    axes[0].set_xlabel("Eigenvalue Rank")
    axes[0].set_ylabel("Curvature Magnitude (Log Scale)")
    
    # Use symlog to cleanly show positive, zero, and negative eigenvalues on a log scale
    axes[0].set_yscale('symlog', linthresh=1e-10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Plot B: Eigenvector Composition of the most problematic directions
    # We plot the composition of the smallest/most negative eigenvalue (Rank 1)
    # and the largest eigenvalue (Rank 7)
    width = 0.35
    x_idx = np.arange(num_params)
    
    axes[1].bar(x_idx - width/2, evecs_np[:, 0]**2 * 100, width, label=f"Min Curvature ($\lambda_1 = {evals_np[0]:.1e}$)", color='red' if evals_np[0] < 0 else 'lightblue')
    axes[1].bar(x_idx + width/2, evecs_np[:, -1]**2 * 100, width, label=f"Max Curvature ($\lambda_7 = {evals_np[-1]:.1e}$)", color='darkblue')
    
    axes[1].set_title("Parameter Composition of Curvature Extremes")
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(param_names)
    axes[1].set_ylabel("Fractional Composition (%)")
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# Execution:
# analyze_hessian_topology(data_chunks, t_gps, r_gps, v_gps, tle_base, center_freq, epoch_unix, num_passes=3)

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.func import grad, hessian
import dsgp4
import diffod.state as state
import diffod.functional.system as system
from diffod.utils import unix_to_mjd
from astropy.time import Time

def plot_hessian_eigenspace_stability(data_chunks, tle_base, center_freq, epoch_unix, num_passes=3):
    """
    Computes the exact Hessian of the Doppler RMSE, evaluates its eigenspectrum 
    to find saddle points (negative eigenvalues) and null-spaces (near-zero), 
    and plots the 'Danger Vectors' that cause filter divergence.
    """
    param_names = ["n", "f", "g", "h", "k", "L"]
    num_params = len(param_names)
    chunk = data_chunks[0]
    
    # --- 1. Setup ---
    active_pass_ids = chunk["pass_ids"][:num_passes]
    pass_mask = torch.isin(chunk["c"], active_pass_ids)
    
    t_active = chunk["t"][pass_mask]
    d_active_true = chunk["d_true"][pass_mask]
    t_mean_window = float(torch.mean(t_active))
    
    init_tle, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
    eval_ssv = state.MEE_SSV(init_tle=init_tle, num_measurements=len(t_active), fit_bstar=False)
    eval_ssv.add_linear_bias(name="pass_freq_bias", group_indices=chunk["c"][pass_mask])
    x_eval = eval_ssv.get_initial_state()
    
    station_model = system.DifferentiableStation(
        lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, ref_unix=t_mean_window, ref_gmst_rad=0.0
    )
    prop_eval = system.SGP4(ssv=eval_ssv)
    meas_eval = system.DopplerMeasurement(
        ssv=eval_ssv, station_model=station_model, freq_bias_group=eval_ssv.get_bias_group("pass_freq_bias"), time_bias_group=None
    )
    pipe_eval = system.MeasurementPipeline(propagator=prop_eval, measurement_model=meas_eval)
    t_since_meas = (t_active - t_mean_window) + 0.277
    
    # --- 2. Define Doppler RMSE Objective ---
    def loss_doppler(x):
        d_pred = pipe_eval(x=x, tsince=t_since_meas, epoch=t_mean_window, center_freq=center_freq)
        return torch.sqrt(torch.mean((d_active_true - d_pred)**2))

    # --- 3. Compute Gradient and Hessian ---
    g_dopp = grad(loss_doppler)(x_eval)[:num_params]
    H_dopp = hessian(loss_doppler)(x_eval)[:num_params, :num_params]
    
    # Calculate Newton-scaled sensitivity: |g| / |diag(H)|
    # This represents the magnitude of the theoretical parameter update
    h_diag_abs = torch.abs(torch.diag(H_dopp))
    newton_sensitivity = (torch.abs(g_dopp) / (h_diag_abs + 1e-16)).detach().cpu().numpy()
    
    # --- 4. Scale Hessian to Dimensionless Curvature Matrix ---
    # We pre- and post-multiply by D^{-1/2} to remove unit scaling issues
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(h_diag_abs + 1e-16))
    H_norm = D_inv_sqrt @ H_dopp @ D_inv_sqrt
    
    # --- 5. Eigendecomposition ---
    evals, evecs = torch.linalg.eigh(H_norm)
    evals_np = evals.detach().cpu().numpy()
    evecs_np = evecs.detach().cpu().numpy()
    
    # Identify Danger Vectors: Eigenvalues <= 0.05 (Saddle points and flat valleys)
    danger_threshold = 0.05
    danger_indices = np.where(evals_np < danger_threshold)[0]
    
    # --- 6. Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot A: Newton-Scaled Sensitivity and Eigenvalue Spectrum
    x_axis = np.arange(num_params)
    axes[0].bar(x_axis - 0.2, newton_sensitivity / np.max(newton_sensitivity), 0.4, label="Scaled Sensitivity $(|g_i| / |H_{ii}|)$", color="#1f77b4")
    
    # Map eigenvalues to colors (Red for Danger/Negative, Green for Stable/Positive)
    colors = ['#d62728' if val < danger_threshold else '#2ca02c' for val in evals_np]
    axes[0].bar(x_axis + 0.2, evals_np, 0.4, label="Hessian Eigenvalues $(\lambda)$", color=colors)
    
    axes[0].set_title(f"Optimization Landscape Stability ({num_passes} Passes)")
    axes[0].set_xticks(x_axis)
    axes[0].set_xticklabels(param_names)
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Plot B: Composition of the Danger Subspace (Null-space & Saddle Points)
    if len(danger_indices) > 0:
        danger_composition = (evecs_np[:, danger_indices]**2).T
        sns.heatmap(
            danger_composition, 
            annot=True, cmap="Reds", fmt=".2f",
            xticklabels=param_names,
            yticklabels=[f"$\lambda_{i} = {evals_np[i]:.2e}$" for i in danger_indices],
            ax=axes[1]
        )
        axes[1].set_title(f"Divergent / Null-Space Composition ($\lambda < {danger_threshold}$)")
        axes[1].set_xlabel("Modified Equinoctial Elements")
        axes[1].set_ylabel("Danger Eigenvectors")
    else:
        axes[1].text(0.5, 0.5, "Strictly Convex Landscape\n(No Danger Vectors)", ha='center', va='center', fontsize=14)
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()