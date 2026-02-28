import torch
import matplotlib.pyplot as plt
import dsgp4
import numpy as np
import torch
import matplotlib.pyplot as plt

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

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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