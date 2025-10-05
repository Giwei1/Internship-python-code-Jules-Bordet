import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
import matplotlib.cm as cm





def load_data():
    df = pd.read_csv("AuditLog_p1_0704Modified.csv")
    df_motion = pd.read_csv("p1_0704_sample1_MClimited.csv")
    return df, df_motion


def detect_end_of_motion(t, z_raw, tolerance=2, stable_duration=1):
    """
    This function is used to detect the end of the tumor motion.
    When the simulator finishes reproducing the motion, it stops moving — this can be clearly seen in the MRI measurements.
    This makes it easier to align the two signals.
    However, if the motion stops prematurely or if the MRI irradiation ends before the motion is completed, there will be no steady segment at the end, and therefore this method cannot be used.
    """
    sampling_interval = np.median(np.diff(t))
    window_size = int(stable_duration / sampling_interval)
    stable_time_start = None
    for i in range(len(z_raw) - window_size, 0, -1):
        segment = z_raw[i:i + window_size]
        if np.max(segment) - np.min(segment) <= 2 * tolerance:
            stable_time_start = t[i]
        else:
            if stable_time_start is not None:
                break
    if stable_time_start is None:
        stable_time_start = t[-1]
    return stable_time_start


def align_motion_data(t, t_motion, z_raw,x_raw, y_raw,  inf_sup, stable_time_start):
    """
    Extracts and aligns the useful portions of the MRI and simulator motion signals.

    Steps:
    1. Selects the useful part of the MRI signal (before the stable phase).
    2. Extracts the corresponding duration from the simulator signal.
    3. Aligns both signals in time based on their start points.
    4. Plots the aligned signals (MRI vs. Simulator) for visual verification.

    Returns:
        t_z_cut (array): Time values of the useful MRI segment.
        z_raw_cut (array): MRI z-position during the useful period.
        x_raw_cut (array): MRI x-position during the useful period.
        y_raw_cut (array): MRI y-position during the useful period.
        inf_sup_cut_shifted (array): Simulator z-position aligned to MRI.
        t_motion_aligned_shifted (array): Time values aligned with MRI signal.
        mask_motion (array): Boolean mask used to extract the simulator segment.
    """

    mask_z = t <= stable_time_start
    t_z_cut = t[mask_z]
    z_raw_cut = z_raw[mask_z]
    x_raw_cut = x_raw[mask_z]
    y_raw_cut = y_raw[mask_z]
    duration_useful = t_z_cut[-1] - t_z_cut[0]

    t_motion_end = t_motion[-1]
    t_motion_start_new = t_motion_end - duration_useful

    mask_motion = t_motion >= t_motion_start_new
    t_motion_cut = t_motion[mask_motion]
    inf_sup_cut = inf_sup[mask_motion]

    time_shift = t_z_cut[0] - t_motion_cut[0]
    t_motion_aligned = t_motion_cut + time_shift

    inf_sup_cut_shifted = inf_sup_cut[1:]
    t_motion_aligned_shifted = t_motion_aligned[1:]

    # Tracer les signaux retournés pour vérification
    plt.figure(figsize=(12, 6))
    plt.plot(t_z_cut, z_raw_cut, label="z position (MRI)", color='blue', linewidth=1)
    plt.plot(t_motion_aligned_shifted, inf_sup_cut_shifted, label="z position (Simulator)", color='orange',
             linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("z position (mm)")
    plt.title("Aligned Signals: MRI vs Simulator")
    plt.legend()
    plt.grid(True)
    plt.show()

    return t_z_cut, z_raw_cut,x_raw_cut,y_raw_cut, inf_sup_cut_shifted, t_motion_aligned_shifted, mask_motion


def manual_time_shift_motion(t_motion, inf_sup, manual_shift_time=0.0):
    """
    Apply a manual time shift on the Motion signal (simulator)
    """
    return t_motion + manual_shift_time, inf_sup


def optimize_alignment(t_motion, inf_sup, t_ref, z_ref,x_ref, y_ref):
    """
        Aligns the simulator motion signal (t_motion / inf_sup) with the MRI reference signal
        (t_ref / z_ref) by minimizing the RMSE (Root Mean Square Error).

        The function automatically trims the signals to the overlapping time range
        where both data sets are valid.

        Steps:
            1. Iteratively shifts the simulator signal over a given time range (delta_range).
            2. Interpolates the MRI reference signal (z_ref, x_ref, y_ref) onto the shifted
               simulator timeline.
            3. Computes the RMSE between the MRI and simulator z-positions.
            4. Finds the time shift that minimizes the RMSE.
            5. Returns the aligned and trimmed signals, as well as the optimal shift.

        Parameters:
            t_motion (array): Time values of the simulator motion signal.
            inf_sup (array): Simulator z-position (motion) signal.
            t_ref (array): Time values of the MRI reference signal.
            z_ref (array): MRI z-position reference signal.
            x_ref (array): MRI x-position reference signal.
            y_ref (array): MRI y-position reference signal.

        Returns:
            best_t_motion (array): Time values of the aligned and trimmed simulator signal.
            best_inf_sup (array): Corresponding simulator z-position signal after alignment.
            best_t_ref_cut (array): Trimmed MRI time values within the overlap region.
            best_z_ref_cut (array): Trimmed MRI z-position signal (not interpolated).
            best_x_ref_cut (array): Trimmed MRI x-position signal.
            best_y_ref_cut (array): Trimmed MRI y-position signal.
            best_z_interp (array): Interpolated MRI z-position aligned to simulator timeline.
            best_shift (float): Optimal time shift (in seconds) minimizing RMSE.

        Raises:
            RuntimeError: If no overlap between MRI and simulator signals is found.

        Prints:
            The optimal time shift and the corresponding RMSE value.

        """
    delta_range = np.linspace(-16.0, 3.0, 200)
    best_rmse = np.inf
    best_shift = 0
    best_t_motion = None
    best_inf_sup = None
    best_z_interp = None
    best_t_ref_cut = None
    best_z_ref_cut = None

    for delta in delta_range:
        shifted_t_motion = t_motion + delta

        # Interpolation
        z_interp_shifted = np.interp(
            shifted_t_motion, t_ref, z_ref, left=np.nan, right=np.nan
        )
        x_interp_shifted = np.interp(
            shifted_t_motion, t_ref, x_ref, left=np.nan, right=np.nan
        )
        y_interp_shifted = np.interp(
            shifted_t_motion, t_ref, y_ref, left=np.nan, right=np.nan
        )
        mask_valid = ~np.isnan(z_interp_shifted)

        if not np.any(mask_valid):
            continue

        rmse = np.sqrt(np.mean((z_interp_shifted[mask_valid] - inf_sup[mask_valid]) ** 2))

        if rmse < best_rmse:
            best_rmse = rmse
            best_shift = delta
            best_t_motion = shifted_t_motion[mask_valid]
            best_inf_sup  = inf_sup[mask_valid]
            best_z_interp = z_interp_shifted[mask_valid]


            # Cut the non interpolated signal
            t_start = best_t_motion[0]
            t_end   = best_t_motion[-1]
            mask_ref = (t_ref >= t_start) & (t_ref <= t_end)
            best_t_ref_cut = t_ref[mask_ref]
            best_z_ref_cut = z_ref[mask_ref]
            best_x_ref_cut = x_ref[mask_ref]
            best_y_ref_cut = y_ref[mask_ref]

    if best_t_motion is None:
        raise RuntimeError("❌ No optimal shift found.")

    print(f"✅ Optimal time shift: {best_shift:.4f} s with RMSE = {best_rmse:.4f} mm")

    return best_t_motion, best_inf_sup, best_t_ref_cut, best_z_ref_cut,best_x_ref_cut, best_y_ref_cut, best_z_interp, best_shift


def plot_signals(t_aligned, z_interp, inf_sup,facteurZ = 1):
    """Plot both signals on the same graph"""
    plt.figure(figsize=(12, 6))
    plt.plot(t_aligned, z_interp, label="Z position (MRI)", color='blue', linewidth=1)
    plt.plot(t_aligned, facteurZ *inf_sup, label="Z position (simulator)", color='orange', linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Z Position (mm)")
    plt.title("MRI vs Simulator - Aligned")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_local_error(t_aligned, z_interp, inf_sup, factor = 1):
    """
        Computes and visualizes the local error between MRI (z_interp) and simulator (inf_sup)
        motion signals, both in absolute values (mm) and as a percentage of the mean peak amplitude.

        The function:
            1. Detects peaks in the MRI signal to estimate the mean motion amplitude.
            2. Computes absolute and relative errors between MRI and simulator signals.
            3. Separately evaluates the total error and the error limited to a specific
               tumor motion range (|tumor| ≤ 3 mm).
            4. Displays:
                - A two-panel figure showing error evolution over time (mm and %).
                - A histogram showing the distribution of relative errors.

        Parameters:
            t_aligned (array): Aligned time values common to both MRI and simulator signals.
            z_interp (array): Interpolated MRI z-position signal.
            inf_sup (array): Simulator z-position signal.
            factor (float, optional): Scaling factor applied to the simulator signal
                                      (default = 1).

        Computed Variables:
            mean_peaks (float): Mean amplitude of detected MRI peaks.
            error_full (array): Absolute error between MRI and simulator signals.
            error_percent (array): Error expressed as a percentage of the mean peak value.
            error_lim (array): Error limited to tumor positions within ±3 mm.
            error_lim_percent (array): Limited error expressed in percentage.

        Plots:
            - Two stacked time-series bar/line plots (absolute and relative error).
            - A histogram of error distributions (total and limited error regions).

        Notes:
            The red dashed line at 100% indicates an error equal to the mean peak value.
            Useful for identifying deviations in synchronization or signal reconstruction.

        """
    #calculate mean peaks value
    peaks, _ = find_peaks(abs(z_interp))
    mean_peaks = np.mean(np.abs(z_interp[peaks]))
    inf_sup = inf_sup*factor
    error_full = np.abs(z_interp - inf_sup)

    # Error in % in function of mean peaks
    # Error in +-3mm inteval
    mask_lim = (inf_sup >= -3) & (inf_sup <= 3)
    error_lim = np.where(mask_lim, error_full, 0)
    error_percent = (error_full / mean_peaks) * 100
    error_lim_percent = (error_lim / mean_peaks) * 100
    mean_error = np.mean(error_full)
    total_error = np.sum(error_full)
    mean_error_percent = (mean_error / mean_peaks) * 100

    # --- Figure with 2 plots---
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

    # --- Up : complete error ---
    ax1 = axes[0]
    ax1.bar(t_aligned, error_full, width=0.08, color='green', alpha=0.6, label='absolute error (mm)')
    ax1.set_ylabel("Error (mm)", color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(t_aligned, error_percent, color='orange', linewidth=1.5, label=' relative error (%)')
    ax1_twin.set_ylabel("Error (%)", color='orange')
    ax1_twin.tick_params(axis='y', labelcolor='orange')
    ax1_twin.axhline(100, color='red', linestyle='--', linewidth=1.5, label='100% error')

    ax1.set_title("Local error between phantom and MRI (mm and % of mean peaks)")
    ax1.grid(True)

    # --- bottom : limited error ---
    ax2 = axes[1]
    ax2.bar(t_aligned, error_lim, width=0.08, color='green', alpha=0.6,
            label=f'limited error (|tumor| ≤ {3} mm, mm)')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (mm)", color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax2_twin = ax2.twinx()
    ax2_twin.plot(t_aligned, error_lim_percent, color='orange', linewidth=1.5, label='Limited error (%)')
    ax2_twin.set_ylabel("Error (%)", color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    ax2_twin.axhline(100, color='red', linestyle='--', linewidth=1.5, label='100% error')

    ax2.set_title(f"Local error filtered (|tumor| ≤ {3} mm) - mm and % of mean peaks")
    ax2.grid(True)


    stats_text = (
        f"Mean peak value: {mean_peaks}"
    )
    ax1.text(
        0.02, 0.95, stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )
    fig.tight_layout()
    plt.show()

    # second graph more understandable
    # --- Histogram  ---
    bins = np.linspace(0, np.max(error_percent), 50)
    counts_full, edges = np.histogram(error_percent, bins=bins)
    counts_lim, _ = np.histogram(error_lim_percent, bins=bins)

    percent_full = counts_full / counts_full.sum() * 100
    percent_lim = counts_lim / counts_lim.sum() * 100
    bin_centers = (edges[:-1] + edges[1:]) / 2

    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, percent_full, width=np.diff(edges), align='center', color='purple', alpha=0.7,
            label='Total error ')
    plt.bar(bin_centers, percent_lim, width=np.diff(edges), align='center', color='cyan', alpha=0.5,
            label='Error in limited interval (|tumor| ≤ 3 mm)')
    plt.xlabel("Error (% of mean peak value)")
    plt.ylabel("% of points")
    plt.title("Distribution of errors between predicted and actual position (in %)")
    plt.grid(True)
    plt.axvline(100, color='red', linestyle='--', label='100% error')
    plt.legend()

    # 🔹 box with extra information
    stats_text = (
        f"Mean peak value : {mean_peaks:.2f} mm\n"
        f"Number of points in total error : {len(error_full)}\n"
        f"Number of points in limited error : {np.count_nonzero(error_lim)}"
    )

    plt.text(0.98, 0.95, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    plt.show()


def compute_beam_stats(df, t):
    """
        Computes beam-on statistics from LINAC and BEAM HOLD states over time.

        The function analyzes the treatment log to determine:
            - When the LINAC was active.
            - How long the beam was actually irradiating (not held).
            - The percentage of irradiation time relative to the total active period.

        Steps:
            1. Identify periods where LINAC is active ("Active").
            2. Determine time intervals where the beam is not held ("DontHoldBeam").
            3. Calculate total active time, effective irradiation time, and their ratio.
            4. Return and print beam-on statistics.

        Parameters:
            df (DataFrame): DataFrame containing "LINAC State" and "BEAM HOLD State" columns.
            t (array): Time values (in seconds) corresponding to the DataFrame entries.

        Returns:
            irradiating (array of bool): Boolean array indicating beam-on periods.
            beam_percent_real (float): Percentage of effective irradiation time
                                       relative to total LINAC active time.
            stats_text (str): Formatted text summary of computed statistics.

        Raises:
            ValueError: If the LINAC was never active in the provided data.

        Prints:
            - Total duration from LINAC active (s)
            - Effective irradiation time (s)
            - Irradiation percentage (%)

        """
    linac_state = df["LINAC State"].values
    beam_state = df["BEAM HOLD State"].values
    irradiating = (linac_state == "Active") & (beam_state == "DontHoldBeam")
    active_indices = np.where(linac_state == "Active")[0]

    if len(active_indices) == 0:
        raise ValueError("LINAC was never active.")

    start_time_active = t[active_indices[0]]
    mask_after_active = t >= start_time_active
    t_active = t[mask_after_active]
    irradiating_active = irradiating[mask_after_active] # True False list only when LINAC active
    total_time_active = t_active[-1] - t_active[0]

    # Calculate real beam-on time
    beam_time_real = 0.0 #the real time the beam was on during LINAC active

    for i in range(1, len(t_active)):
        if irradiating_active[i - 1]:
            dt = t_active[i] - t_active[i - 1]
            beam_time_real += dt

    beam_percent_real = (beam_time_real / total_time_active) * 100

    print("\n🔎 Beam statistics:")
    print(f"Total duration from LINAC active: {total_time_active:.2f} s")
    print(f"Effective irradiation time: {beam_time_real:.2f} s")
    print(f"➡️ Irradiation percentage: {beam_percent_real:.2f} %")


    stats_text = (
        f"Total active: {total_time_active:.2f} s\n"
        f"Irradiation: {beam_time_real:.2f} s\n"
        f"Percent: {beam_percent_real:.2f} %"
    )

    return irradiating, beam_percent_real, stats_text

def plot_irradiation_with_position(
    t_aligned,
    x_signal, y_signal, z_signal,
    irradiating_aligned,
    beam_percent_real,
    stats_text,
    bins=50
):
    """
        Visualizes 3D tumor position signals (X, Y, Z) and their norm, highlighting irradiation periods.

        The function produces a 2x2 figure that:
            1. Plots the X, Y, and Z position signals over time, highlighting irradiation periods in green.
            2. Plots the vector norm √(X² + Y² + Z²) over time with irradiation shading.
            3. Displays the time distribution  of each spatial component (X, Y, Z) and
               indicates how much of that time was irradiated.
            4. Shows the time distribution of the position norm with the percentage of irradiated time.
            5. Adds textual annotations showing beam statistics and colorbar normalization.

        Parameters:
            t_aligned (array): Time axis aligned with all motion signals (in seconds).
            x_signal (array): X-axis motion data (mm).
            y_signal (array): Y-axis motion data (mm).
            z_signal (array): Z-axis motion data (mm).
            irradiating_aligned (array of bool): Boolean array indicating when the beam was ON.
            beam_percent_real (float): Effective irradiation percentage relative to total active time.
            stats_text (str): Text summary of beam statistics (displayed in the figure).
            bins (int, optional): Number of bins for the density estimation. Default is 50.

        Returns:
            None

        Notes:
            - Green shaded areas correspond to periods when irradiation was active.
            - plots on the right show the distribution of
              positions and how often each amplitude was irradiated.
            - A colorbar indicates the proportion of irradiated time for each position range.

        Visualization:
            Top left: X, Y, Z positions with irradiation periods.
            Bottom left: Norm of position vector with irradiation periods.
            Top right: Density plots of X, Y, Z with irradiation ratios.
            Bottom right: Density plot of norm with irradiation ratios.

        """
    # 🔹 Harmoniser les tailles (évite IndexError)
    n = min(len(t_aligned), len(x_signal), len(y_signal), len(z_signal), len(irradiating_aligned))
    t_aligned = t_aligned[:n]
    x_signal = x_signal[:n]
    y_signal = y_signal[:n]
    z_signal = z_signal[:n]
    irradiating_aligned = irradiating_aligned[:n]
    # Norm calculation
    norm_signal = np.sqrt(x_signal**2 + y_signal**2 + z_signal**2)

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(14, 10),
        gridspec_kw={'width_ratios': [3, 1]}, sharex='col'
    )

    # -------------------
    # 1) Top graphic : X, Y, Z with irradiation
    ax_xyz = axes[0, 0]
    ax_xyz.plot(t_aligned, x_signal, label="X", color="red", linewidth=1)
    ax_xyz.plot(t_aligned, y_signal, label="Y", color="green", linewidth=1)
    ax_xyz.plot(t_aligned, z_signal, label="Z", color="blue", linewidth=1)

    # Adding of irradiation zones
    start_time = None
    for i in range(1, len(t_aligned)):
        if irradiating_aligned[i] and not irradiating_aligned[i - 1]:
            start_time = t_aligned[i]
        if not irradiating_aligned[i] and irradiating_aligned[i - 1] and start_time is not None:
            end_time = t_aligned[i]
            ax_xyz.axvspan(start_time, end_time, color='green', alpha=0.3)
            start_time = None
    if start_time is not None and irradiating_aligned[-1]:
        ax_xyz.axvspan(start_time, t_aligned[-1], color='green', alpha=0.3)

    ax_xyz.set_ylabel("Position (mm)")
    ax_xyz.set_title("Positions X, Y, Z with irradiation (green = ON)")
    ax_xyz.grid(True)
    ax_xyz.legend()

    # -------------------
    # 2) Bottom left graphic: Norm with irradiation
    ax_norm = axes[1, 0]
    ax_norm.plot(t_aligned, norm_signal, label="Norm √(X²+Y²+Z²)", color="purple", linewidth=1)

    #  Irradiation zones
    start_time = None
    for i in range(1, len(t_aligned)):
        if irradiating_aligned[i] and not irradiating_aligned[i - 1]:
            start_time = t_aligned[i]
        if not irradiating_aligned[i] and irradiating_aligned[i - 1] and start_time is not None:
            ax_norm.axvspan(start_time, t_aligned[i], color='green', alpha=0.3)
            start_time = None
    if start_time is not None and irradiating_aligned[-1]:
        ax_norm.axvspan(start_time, t_aligned[-1], color='green', alpha=0.3)

    ax_norm.set_xlabel("Time (s)")
    ax_norm.set_ylabel("Norm (mm)")
    ax_norm.set_title("Vector norm (√X²+Y²+Z²) with irradiation")
    ax_norm.grid(True)
    ax_norm.legend()

    # -------------------
    # Intern function to trace the density with irradiation
    def plot_density_with_irradiation(ax, sig, sig_irr, label, col):
        kde_all = gaussian_kde(sig)
        y_vals = np.linspace(sig.min(), sig.max(), 300)
        density_all = kde_all(y_vals)

        kde_irr = gaussian_kde(sig_irr) if len(sig_irr) > 10 else None
        density_irr = kde_irr(y_vals) if kde_irr else np.zeros_like(density_all)

        # Ratio irradiation / presence
        ratio = np.divide(density_irr, density_all, out=np.zeros_like(density_all), where=density_all > 0)

        #  KDE curve normal (% of total time)
        percent_time = density_all / density_all.sum() * 100
        ax.plot(percent_time, y_vals, color=col, linewidth=2, label=label)

        #Background gradient
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.colormaps["Reds"]
        for i in range(len(y_vals) - 1):
            ax.axhspan(
                y_vals[i], y_vals[i + 1],
                xmin=0, xmax=1,
                facecolor=cmap(norm(ratio[i])),
                alpha=0.4, edgecolor="none"
            )

    # -------------------
    # 3) Top right : X/Y/Z
    ax_density_xyz = axes[0, 1]
    for sig, label, col in zip(
            [x_signal, y_signal, z_signal],
            ["X", "Y", "Z"],
            ["red", "green", "blue"]
    ):

        sig_irr = sig[irradiating_aligned]  # positions irradiées uniquement
        plot_density_with_irradiation(ax_density_xyz, sig, sig_irr, label, col)

    ax_density_xyz.set_xlabel("% Time ")
    ax_density_xyz.set_ylabel("Amplitude (mm)")
    ax_density_xyz.set_title("Time distribution X/Y/Z with % irradiation")
    ax_density_xyz.grid(True)
    ax_density_xyz.legend()

    # -------------------
    # 4) Bottom right : Norm
    ax_density_norm = axes[1, 1]
    sig_irr = norm_signal[irradiating_aligned]
    plot_density_with_irradiation(ax_density_norm, norm_signal, sig_irr, "Norme", "purple")

    ax_density_norm.set_xlabel("% Time")
    ax_density_norm.set_ylabel("Amplitude (mm)")
    ax_density_norm.set_title("Time distribution Norm with % irradiation")
    ax_density_norm.grid(True)
    ax_density_norm.legend()

    # -------------------
    # Global annotations
    fig.text(
        0.98, 0.95, f"Irradiation: {beam_percent_real:.2f} %",
        fontsize=12, va='top', ha='right',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )
    fig.text(
        0.98, 0.05, stats_text,
        fontsize=10, va='bottom', ha='right',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )


    # Adding of a colorbar at the right
    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.colormaps["Reds"]

    cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # [x, y, largeur, hauteur]
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("% irradiated time / total time in this position")
    plt.subplots_adjust(right=0.9)  # laisse de la place à droite pour la colorbar
    plt.show()

def rescale_motion_signal(signal, original_peak=10.0, target_peak=6.0):
    """
    Rescales the amplitude of a motion signal without modifying the original source file.
    Example: if the peaks were 10 mm but should be 6 mm.

    Args:
        signal : np.array
            Original motion signal (Inf/Sup, AP, LR, etc.)
        original_peak : float
            Expected maximum amplitude (e.g., 10 mm)
        target_peak : float
            Desired amplitude (e.g., 6 mm)

    Returns:
        signal_rescaled : np.array
            Rescaled motion signal
    """

    factor = target_peak / original_peak
    return signal * factor


def plot_position_vs_prediction(t_aligned, x_ref, y_ref, z_ref,
    x_pred, y_pred, z_pred,
    x_err, y_err, z_err,
    smooth_window=21, poly_order=3):
    """
       Plots real vs predicted 3D positions (Z and vector norm) and shows relative error distributions.

       The function creates a 2x2 figure:
           1. Top-left: Z position – real (MRI) vs prediction.
           2. Top-right: Cumulative relative error as a function of Z amplitude.
           3. Bottom-left: Norm of position vector √(X²+Y²+Z²) – real vs prediction.
           4. Bottom-right: Cumulative relative error as a function of norm amplitude.

       Additionally, it produces a histogram showing the distribution of relative errors
       for Z and the norm signals, expressed as a percentage of the mean peak value.

       Parameters:
           t_aligned (array): Aligned time values (s).
           x_ref, y_ref, z_ref (array): Reference (real) motion signals (mm).
           x_pred, y_pred, z_pred (array): Predicted motion signals (mm).
           x_err, y_err, z_err (array): Absolute error signals (mm).
           smooth_window (int, optional): Window size for optional smoothing (default 21).
           poly_order (int, optional): Polynomial order for optional smoothing (default 3).

       Returns:
           None

       Notes:
           - Relative errors are normalized by the total absolute error or by the mean peak amplitude.
           - Histograms show the relative error distribution for each axis and the norm.
           - The function automatically handles NaNs/Inf and aligns all arrays to the same length.
           - Figures include textual annotations for total error and mean peak values.
       """

    mask = (t_aligned > 25) & (t_aligned < 415)
    t_aligned = t_aligned[mask]
    x_ref = x_ref[mask];
    y_ref = y_ref[mask];
    z_ref = z_ref[mask]
    x_pred = x_pred[mask];
    y_pred = y_pred[mask];
    z_pred = z_pred[mask]
    x_err = x_err[mask];
    y_err = y_err[mask];
    z_err = z_err[mask]

    # --- Cleaning of NaN/inf ---
    mask_valid = np.isfinite(x_ref) & np.isfinite(y_ref) & np.isfinite(z_ref) & \
                 np.isfinite(x_pred) & np.isfinite(y_pred) & np.isfinite(z_pred) & \
                 np.isfinite(x_err) & np.isfinite(y_err) & np.isfinite(z_err)

    t_aligned = t_aligned[mask_valid]
    x_ref = x_ref[mask_valid];
    y_ref = y_ref[mask_valid];
    z_ref = z_ref[mask_valid]
    x_pred = x_pred[mask_valid];
    y_pred = y_pred[mask_valid];
    z_pred = z_pred[mask_valid]
    x_err = x_err[mask_valid];
    y_err = y_err[mask_valid];
    z_err = z_err[mask_valid]

    # --- egalise all lengths ---
    n = min(len(t_aligned), len(x_ref), len(y_ref), len(z_ref),
            len(x_pred), len(y_pred), len(z_pred),
            len(x_err), len(y_err), len(z_err))
    t_aligned = t_aligned[:n]
    x_ref, y_ref, z_ref = x_ref[:n], y_ref[:n], z_ref[:n]
    x_pred, y_pred, z_pred = x_pred[:n], y_pred[:n], z_pred[:n]
    x_err, y_err, z_err = x_err[:n], y_err[:n], z_err[:n]

    # --- Cleaning  ---
    mask_valid = np.isfinite(x_ref) & np.isfinite(y_ref) & np.isfinite(z_ref) & \
                 np.isfinite(x_pred) & np.isfinite(y_pred) & np.isfinite(z_pred) & \
                 np.isfinite(x_err) & np.isfinite(y_err) & np.isfinite(z_err)
    t_aligned = t_aligned[mask_valid]
    x_ref, y_ref, z_ref = x_ref[mask_valid], y_ref[mask_valid], z_ref[mask_valid]
    x_pred, y_pred, z_pred = x_pred[mask_valid], y_pred[mask_valid], z_pred[mask_valid]
    x_err, y_err, z_err = x_err[mask_valid], y_err[mask_valid], z_err[mask_valid]
    z_pred_smooth = z_pred

    # --- Norms ---
    norm_ref = np.sqrt(x_ref ** 2 + y_ref ** 2 + z_ref ** 2)
    norm_pred = np.sqrt(x_pred ** 2 + y_pred ** 2 + z_pred_smooth ** 2)
    norm_err = np.sqrt(x_err ** 2 + y_err ** 2 + z_err ** 2)
    total_err = np.sum(norm_err)
    total_z_err = np.sum(np.abs(z_err))

    # --- Relatives errors (%) ---
    z_rel_err = np.abs(z_err / total_z_err) * 100
    norm_rel_err = np.abs(norm_err / total_err) * 100

    # --- Figure 2x2 ---
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(14, 10),
        gridspec_kw={'width_ratios': [3, 1]}, sharex='col'
    )

    # --- Top left : Z real vs prediction ---
    ax_z = axes[0, 0]
    ax_z.plot(t_aligned, z_ref, label="IRM (real)", color="blue", linewidth=1.5)
    ax_z.plot(t_aligned, z_pred_smooth, label="Prediction", color="orange", linestyle="--", linewidth=1.5)
    ax_z.set_ylabel("Position Z (mm)")
    ax_z.set_title("Position Z : Real vs Prediction")
    ax_z.legend()
    ax_z.grid(True)

    # --- Top right : Relative Z position error  ---
    ax_err_z = axes[0, 1]

    # Cutting Z amplitude in intervals
    bins_z = np.linspace(z_ref.min(), z_ref.max(), 70)
    digitized_z = np.digitize(z_ref, bins_z)

    # Sum of errors in each interval
    sum_err_by_amp_z = [z_rel_err[digitized_z == i].sum() for i in range(1, len(bins_z))]
    centers_z = 0.5 * (bins_z[:-1] + bins_z[1:])

    # tracing the errors for each interval
    ax_err_z.plot(sum_err_by_amp_z, centers_z, "-o", color="red")

    ax_err_z.set_xlabel("Sum of relative errors (% of total error)")
    ax_err_z.set_ylabel("Amplitude Z (mm)")
    ax_err_z.set_title("relative error cumulated for each amplitude")
    ax_err_z.grid(True)
    ax_err_z.text(0.95, 0.95, f"total error Z = {total_z_err:.2f} mm",
                  ha="right", va="top", transform=ax_err_z.transAxes,
                  fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # --- Bottom left :Real norm vs prediction ---
    ax_norm = axes[1, 0]
    ax_norm.plot(t_aligned, norm_ref, label="IRM (norm, mm)", color="purple", linewidth=1.5)
    ax_norm.plot(t_aligned, norm_pred, label="Prediction (norm, mm)", color="orange", linestyle="--", linewidth=1.5)
    ax_norm.set_xlabel("Time (s)")
    ax_norm.set_ylabel("Norm (mm)")
    ax_norm.set_title("Norm √(X²+Y²+Z²) : Real vs Prediction")
    ax_norm.legend()
    ax_norm.grid(True)

    # --- Bottom right : Cumulated relative error for each interval norm ---
    ax_err_norm = axes[1, 1]

    # Cutting the norm interval in a lot of intervals for more continuity
    bins_norm = np.linspace(norm_ref.min(), norm_ref.max(), 100)
    digitized_norm = np.digitize(norm_ref, bins_norm)

    # Sum of errors
    sum_err_by_amp_norm = [norm_rel_err[digitized_norm == i].sum() for i in range(1, len(bins_norm))]
    centers_norm = 0.5 * (bins_norm[:-1] + bins_norm[1:])

    # Tracing
    ax_err_norm.plot(sum_err_by_amp_norm, centers_norm, "-o", color="red")

    ax_err_norm.set_xlabel("Sum of relative errors (% of total error)")
    ax_err_norm.set_ylabel("Amplitude Norm (mm)")
    ax_err_norm.set_title("relative error cumulated for each amplitude (Norme)")
    ax_err_norm.grid(True)
    ax_err_norm.text(0.95, 0.95, f"Total error Norm = {total_err:.2f} mm",
                     ha="right", va="top", transform=ax_err_norm.transAxes,
                     fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    plt.tight_layout()
    plt.show()



    # --- Histogram of relatives errors ---
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))

    # Calcul of mean peaks for Z
    peaks, _ = find_peaks(np.abs(z_ref))
    mean_peak_z = np.mean(np.abs(z_ref[peaks]))

    #Relative error in % compared to the mean of the peaks
    z_rel_err = np.abs(z_err) / mean_peak_z * 100

    # For the norm, you can do the same with the mean value of the norm's peaks.
    peaks_norm, _ = find_peaks(norm_ref)
    mean_peak_norm = np.mean(norm_ref[peaks_norm])
    norm_rel_err = np.abs(norm_err) / mean_peak_norm * 100

    # Histogram in percentage of the number of points
    bins = np.linspace(0, max(z_rel_err.max(), norm_rel_err.max()), 50)

    hist_z, bins_z, _ = ax_hist.hist(z_rel_err, bins=bins, density=True, alpha=0.5, label="Z", color="blue")
    hist_norm, bins_n, _ = ax_hist.hist(norm_rel_err, bins=bins, density=True, alpha=0.5, label="Norme", color="orange")

    # Convert to % of points
    hist_z_percent = hist_z * 100 / hist_z.sum()
    hist_norm_percent = hist_norm * 100 / hist_norm.sum()

    ax_hist.cla()  # on nettoie pour re-tracer avec % sur y
    ax_hist.bar(bins[:-1], hist_z_percent, width=np.diff(bins), alpha=0.5, color="blue", label="Z")
    ax_hist.bar(bins[:-1], hist_norm_percent, width=np.diff(bins), alpha=0.5, color="orange", label="Norme")


    ax_hist.set_xlabel("relative error (% of mean peak value)")
    ax_hist.set_ylabel("% of points")
    ax_hist.set_title("Relative Error Distribution (%)")
    ax_hist.legend()
    ax_hist.grid(True, linestyle='--', alpha=0.5)

    #annotad tab
    textstr = '\n'.join((
        f"Total points: {len(z_rel_err)}",
        f"Mean peak value Z: {mean_peak_z:.2f}mm",
        f"Mean peak value Norme: {mean_peak_norm:.2f}mm"
    ))
    ax_hist.text(0.95, 0.95, textstr, transform=ax_hist.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    plt.show()


def main(mode=1, manual_shift=-10.0, original_peak=10.0, target_peak=7.0):
    """
    mode = 1 → Manual shift
    mode = 2 → Automatic alignement with no motion at the end
    mode = 3 → no shifting, only optimize_alignment
    """
    df, df_motion = load_data()
    t = df["t_seconds"].values
    z_raw = df["Z_pos_corr"].values
    x_raw = df["X_pos"].values
    y_raw = df["Y_pos"].values
    t_motion = df_motion["Time"].values
    inf_sup = rescale_motion_signal(
        df_motion["Inf/Sup"].values,
        original_peak=original_peak,
        target_peak=target_peak
    )

    z_pred_corr = df["Z_pred_corr"].values
    t_pred = df["t_seconds"].values

    # #test plot avant modification
    # plt.figure(figsize=(12, 6))
    # plt.plot(t, z_raw, label="Z_smooth (MRI)", color='blue', linewidth=1)
    # plt.plot(t, x_raw, label="X_smooth (MRI)", color='red', linewidth=1)
    # plt.plot(t, y_raw, label="Y_smooth (MRI)", color='green', linewidth=1)
    # plt.xlabel("Temps (s)")
    # plt.ylabel("Position (mm)")
    # plt.title("Signaux X, Y, Z avant coupure")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    if mode == 1:
        print(f"⚠️ Mode 1: décalage manuel de {manual_shift} s")
        # Décalage brut
        t_motion_aligned, inf_sup_aligned = manual_time_shift_motion(
            t_motion, inf_sup, manual_shift
        )
        t_z_cut = t
        z_raw_cut = z_raw
        x_raw_cut = x_raw
        y_raw_cut = y_raw

    elif mode == 2:
        print("⚠️ Mode 2: alignement auto par stabilité MRI")
        stable_time_start = detect_end_of_motion(t, z_raw)
        t_z_cut, z_raw_cut,x_raw_cut,y_raw_cut,  inf_sup_aligned, t_motion_aligned, _ = align_motion_data(
            t, t_motion, z_raw,x_raw,y_raw, inf_sup, stable_time_start
        )

    elif mode == 3:
        print("⚠️ Mode 3: pas de pré-alignement, optimize_alignment seul")
        t_z_cut = t
        z_raw_cut = z_raw
        x_raw_cut = x_raw
        y_raw_cut = y_raw
        t_motion_aligned = t_motion
        inf_sup_aligned = inf_sup

    else:
        raise ValueError("Mode doit être 1, 2 ou 3")

    # Optimisation fine

    t_motion_final, inf_sup_final, t_z_final, z_raw_cut_final,x_raw_cut_final, y_raw_cut_final, z_interp_final, best_shift = optimize_alignment(
        t_motion_aligned, inf_sup_aligned, t_z_cut, z_raw_cut, x_raw_cut, y_raw_cut
    )



    z_pred_interp = np.interp(t_motion_final, t_pred, z_pred_corr, left=np.nan, right=np.nan)

    # Tracés
    plot_signals(t_motion_final, z_interp_final, inf_sup_final,1.2 )
    plot_local_error(t_motion_final, z_interp_final, inf_sup_final,1.2)

    # Stats faisceau
    irradiating, beam_percent_real, stats_text = compute_beam_stats(df, t)
    irradiating_cut = irradiating[t <= t_z_final[-1]]

    plot_irradiation_with_position(
        t_z_final,
        x_raw_cut_final,
        y_raw_cut_final,
        z_raw_cut_final,
        irradiating_cut,
        beam_percent_real,
        stats_text
    )
    plot_position_vs_prediction(
        t,
        x_raw, y_raw, z_raw_cut,  # signaux réels
        df["X_pred"].values, df["Y_pred"].values, z_pred_corr,  # prédictions
        df["X_error"].values, df["Y_error"].values, df["Z_error"].values  # erreurs
    )


if __name__ == "__main__":
    main(mode=1, manual_shift=-8.0)





# def compute_latency_mri(z_ref_time, z_ref_signal, irradiating, t_linac, threshold_mm=3):
#     """
#     Calcule la latence ON/OFF en utilisant directement le signal IRM (z_ref_cut),
#     déjà découpé/aligné par optimize_alignment.
#
#     Args:
#         z_ref_time : timeline IRM alignée et coupée (t_z_final)
#         z_ref_signal : signal IRM correspondant (z_raw_cut_final)
#         irradiating : état du faisceau (True/False)
#         t_linac : timeline du faisceau
#         threshold_mm : seuil en mm pour décider ON/OFF
#     """
#     # Fenêtre commune
#     t_start, t_end = z_ref_time[0], z_ref_time[-1]
#     mask_beam = (t_linac >= t_start) & (t_linac <= t_end)
#     t_beam = t_linac[mask_beam]
#     irr_beam = irradiating[mask_beam].astype(int)
#
#     if len(t_beam) < 2:
#         print("Pas assez de données faisceau dans la fenêtre commune.")
#         return
#
#     # Détecter transitions ON et OFF
#     on_edges_idx = np.where((irr_beam[1:] == 1) & (irr_beam[:-1] == 0))[0] + 1
#     off_edges_idx = np.where((irr_beam[1:] == 0) & (irr_beam[:-1] == 1))[0] + 1
#     beam_on_times = t_beam[on_edges_idx]
#     beam_off_times = t_beam[off_edges_idx]
#
#     # Détecter les instants "attendus" via IRM
#     inside_mask = np.abs(z_ref_signal) <= threshold_mm
#     inside_edges = np.where((inside_mask[1:] == 1) & (inside_mask[:-1] == 0))[0] + 1
#     outside_edges = np.where((inside_mask[1:] == 0) & (inside_mask[:-1] == 1))[0] + 1
#     expected_on_times = z_ref_time[inside_edges]
#     expected_off_times = z_ref_time[outside_edges]
#     # Chercher première occurrence après chaque transition
#     latency_on = []
#     for t_on in beam_on_times:
#         # Chercher l'entrée la plus proche (avant ou après)
#         idx = np.argmin(np.abs(expected_on_times - t_on)) if expected_on_times.size > 0 else None
#         if idx is not None:
#             t_expected = expected_on_times[idx]
#             latency_on.append(t_on - t_expected)
#
#     latency_off = []
#     for t_off in beam_off_times:
#         idx = np.argmin(np.abs(expected_off_times - t_off)) if expected_off_times.size > 0 else None
#         if idx is not None:
#             t_expected = expected_off_times[idx]
#             latency_off.append(t_off - t_expected)
#
#     # Reporting
#     print("\n=== LATENCY (MRI reference) ===")
#     print(f"Number of ON transitions: {len(latency_on)}")
#     print(f"Number of OFF transitions: {len(latency_off)}")
#     if latency_on:
#         print(f"Mean latency ON:  {np.mean(latency_on)  :.2f} ms")
#     else:
#         print("Mean latency ON:  No ON transitions detected")
#     if latency_off:
#         print(f"Mean latency OFF: {np.mean(latency_off) :.2f} ms")
#     else:
#         print("Mean latency OFF: No OFF transitions detected")
#
#     # Histogrammes
#     if latency_on or latency_off:
#         plt.figure(figsize=(12, 5))
#         if latency_on:
#             plt.hist(np.array(latency_on) , bins=30, alpha=0.7, label="Latency ON (ms)")
#         if latency_off:
#             plt.hist(np.array(latency_off) , bins=30, alpha=0.7, label="Latency OFF (ms)")
#         plt.xlabel("Latency (ms)")
#         plt.ylabel("Count")
#         plt.title("Histogram of ON/OFF latencies (MRI reference)")
#         plt.legend()
#         plt.grid(True)
#         plt.show()



#
# def plot_local_error_filtered(t_aligned, z_interp, inf_sup):
#     # Conversion cm → mm
#     z_interp_mm = z_interp
#     inf_sup_mm = inf_sup
#
#     # Masque : position tumeur entre -3 mm et +3 mm
#     mask = (inf_sup_mm >= -3) & (inf_sup_mm <= 3)
#
#     # Erreur uniquement pour les points qui passent le filtre
#     error_mm = np.abs(z_interp_mm - inf_sup_mm)
#     error_mm_filtered = np.where(mask, error_mm, 0)  # 0 si en dehors
#     mean_error_filtered = np.mean(error_mm_filtered) if np.any(mask) else np.nan
#
#
#     # Graphique en barres
#     plt.figure(figsize=(12, 4))
#     plt.bar(t_aligned, error_mm_filtered, width=0.08, color='green', alpha=0.7)
#     plt.xlabel("Time (s)")
#     plt.ylabel("Error (mm)")
#     plt.title("local error (only if the tumor position is between ±3 mm)")
#     plt.grid(True)
#     stats_text = f"Mean error (|pos|<3 mm) = {mean_error_filtered:.2f} mm"
#     plt.text(0.98, 0.95, stats_text,
#              transform=plt.gca().transAxes,
#              fontsize=10, verticalalignment='top', horizontalalignment='right',
#              bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
#     plt.show()
