# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares


# audit_file = r"Logfiles/AuditLog20250508_10RPM.csv"
# audit_file = r"Logfiles/AuditLog20250508_18RPM.csv"
def process_log(path, rpm):
    df_audit = pd.read_csv(path, sep=",", encoding='utf-8')

    # Drop rows where the translation vector is missing or empty
    df_audit = df_audit[df_audit["Translation Vector (x y z)"].notna()]
    df_audit = df_audit[df_audit["Translation Vector (x y z)"].str.strip() != ""]

    # Clean the format: remove parentheses and extra spaces
    df_audit["Translation Vector (x y z)"] = df_audit["Translation Vector (x y z)"].str.replace(r"[()]", "",
                                                                                                regex=True).str.strip()

    # Split into 3 columns and convert to float
    df_audit[["X_pos", "Y_pos", "Z_pos"]] = df_audit["Translation Vector (x y z)"].str.split(expand=True).astype(float)

    df_audit["VectorLength"] = np.sqrt(df_audit["X_pos"] ** 2 + df_audit["Y_pos"] ** 2 + df_audit["Z_pos"] ** 2)
    df_audit = df_audit[df_audit["Z_pos"].notna()]

    df_audit["Record Time Stamp"] = pd.to_datetime(df_audit["Record Time Stamp"])
    df_audit = df_audit.sort_values("Record Time Stamp")
    df_audit["t_seconds"] = (df_audit["Record Time Stamp"] - df_audit["Record Time Stamp"].iloc[0]).dt.total_seconds()
    df_audit["Z_pos_corr"] = df_audit["Z_pos"] - np.mean(df_audit["Z_pos"].values)
    df_audit = df_audit[df_audit["t_seconds"] < (np.max(df_audit["t_seconds"].values) - 10)]

    # Known parameters
    amplitude = 10.0  # known amplitude of sine wave
    cycle_time = 60.0 / rpm  # known cycle time (period) of sine wave
    omega = 2 * np.pi / cycle_time  # angular frequency

    # Fit phase and offset using least squares, assuming amplitude and omega fixed:
    # Model: Z_fit = amplitude * sin(omega * time + phase) + offset
    # We fit phase and offset to minimize difference.

    def model_func(t, phase, offset):
        return amplitude * np.sin(omega * t + phase) + offset

    # Residual function for least squares
    def residuals(params, t, z):
        phase, offset = params
        return z - model_func(t, phase, offset)

    # Initial guess for phase and offset
    initial_guess = [0, np.mean(df_audit["Z_pos_corr"].values)]

    result = least_squares(residuals, initial_guess, args=(df_audit["t_seconds"].values, df_audit["Z_pos_corr"].values))
    phase_fit, offset_fit = result.x

    # Compute fitted sine values
    Z_fit = model_func(df_audit["t_seconds"].values, phase_fit, offset_fit)

    # Compute disagreement (residuals)
    disagreement = df_audit["Z_pos_corr"].values - Z_fit

    # Plot the results
    # plt.figure(figsize=(12,6))

    # plt.subplot(2,1,1)
    # plt.plot(df_audit["t_seconds"].values, df_audit["Z_pos_corr"], label='Measured Z position')
    # plt.plot(df_audit["t_seconds"].values, Z_fit, label='Fitted sine wave', linewidth=2,ls="--")
    # plt.legend()
    # plt.title('Sine Fit to Z Position vs Time')

    # plt.subplot(2,1,2)
    # plt.plot(df_audit["t_seconds"].values, disagreement)
    # plt.title('Disagreement (Measured - Fitted)')
    # plt.xlabel('Time')
    # plt.ylabel('Position Residual')

    # plt.tight_layout()
    # plt.show()

    total_points = len(df_audit["t_seconds"].values)
    total_duration = df_audit["t_seconds"].values[-1] - df_audit["t_seconds"].values[0]
    total_cycles = total_duration / cycle_time

    abs_disagreement = np.abs(disagreement)
    conf_95 = np.percentile(abs_disagreement, 95)

    # plt.figure(figsize=(8,5))
    # plt.hist(abs_disagreement, bins=50, edgecolor='black', alpha=0.7)
    # plt.axvline(conf_95, color='red', linestyle='--', label=f'95% CI = {conf_95:.3f}')
    # plt.title('Histogram of APM accuracy')
    # plt.xlabel('Distance to agreement (mm)')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.grid(True)

    # # Add textbox with info
    # textstr = '\n'.join((
    #     f'Amplitude +/- {amplitude} RPM: {cycle_time*60:.3f}',
    #     f'Total measurement points: {total_points}',
    #     f'Total cycles: {total_cycles:.0f}',
    #     f'95% CI: {conf_95:.3f}',
    # ))

    # # Position textbox in upper right corner
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    # plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
    #                fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=props)

    # plt.show()

    return {
        "df": df_audit,
        "t": df_audit["t_seconds"].values,
        "z_corr": df_audit["Z_pos_corr"].values,
        "Z_fit": Z_fit,
        "disagreement": disagreement,
        "abs_disagreement": abs_disagreement,
        "conf_95": conf_95,
        "total_points": total_points,
        "total_cycles": total_cycles,
        "cycle_time": cycle_time,
        "rpm": rpm
    }


# === CONFIGURATION ===
logfiles = [
    ("AuditLog_0630.csv", 10.0),
    ("AuditLog_0701.csv", 18.0)
]

amplitude = 10.0  # known amplitude of sine wave (mm)

# === PROCESS BOTH FILES ===
results = [process_log(path, rpm) for path, rpm in logfiles]

# === PLOTTING ===

# 1. Measured vs Fitted Z Position
fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex='row')
for i, result in enumerate(results):
    axs[0, i].plot(result["t"], result["z_corr"], label='Measured Z', marker='', lw=1)
    axs[0, i].plot(result["t"], result["Z_fit"], label='Fitted sine', linestyle='--', lw=2)
    axs[0, i].set_title(f'Z Position @ {result["rpm"]:.0f} RPM')
    axs[0, i].set_ylabel("Z Position (mm)")
    axs[0, i].legend()
    axs[0, i].grid(True)

# 2. Disagreement (Residuals)
for i, result in enumerate(results):
    axs[1, i].plot(result["t"], result["disagreement"], lw=1)
    axs[1, i].set_title(f'Residuals @ {result["rpm"]:.0f} RPM')
    axs[1, i].set_xlabel("Time (s)")
    axs[1, i].set_ylabel("Residual (mm)")
    axs[1, i].grid(True)

plt.tight_layout()
# fig.savefig("APM Accuracy_residuals20250508.png", dpi=600, bbox_inches='tight')
plt.show()

# 3. Histogram of Absolute Disagreement
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
for i, result in enumerate(results):
    axs[i].hist(result["abs_disagreement"], bins=50, edgecolor='black', alpha=0.7)
    axs[i].axvline(result["conf_95"], color='red', linestyle='--', label='95% CI')
    axs[i].set_title(f'RPM: {result["rpm"]:.0f}', fontsize=14)
    axs[i].set_xlabel('Distance to agreement (mm)', fontsize=14)
    if i == 0:
        axs[i].set_ylabel('Frequency', fontsize=14)
    axs[i].legend()
    axs[i].grid(True)

    # Text box
    textstr = '\n'.join((

        f'Total points: {result["total_points"]}',
        f'Total cycles: {result["total_cycles"]:.0f}',
        f'95% CI: {result["conf_95"]:.3f} mm',
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axs[i].text(0.95, 0.95, textstr, transform=axs[i].transAxes,
                fontsize=11, verticalalignment='top',
                horizontalalignment='right', bbox=props)
fig.suptitle(f"APM Accuracy", fontsize=14, weight='bold')
plt.tight_layout()
# fig.savefig("APM Accuracy20250508.png", dpi=600, bbox_inches='tight')
plt.show()
