import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit
from scipy.signal import find_peaks
from scipy.signal import savgol_filter




def regularisation(path):
    """this function creates a copy of the original csv file and modifies it partially to make it
    easier to work with the data"""

    input_path, output_path = path
    # Read the original CSV file
    df_original = pd.read_csv(input_path, sep=",", encoding='utf-8', low_memory=False)

    # Create a copy of the original CSV file to modify it
    df_modified = df_original.copy()

    # Modify values in the copied DataFrame
    # Drop rows where the translation vector is missing or empty
    df_modified = df_modified[df_modified["Translation Vector (x y z)"].notna()]
    df_modified = df_modified[df_modified["Translation Vector (x y z)"].str.strip() != ""]

    # Clean the format: remove parentheses and extra spaces
    df_modified["Translation Vector (x y z)"] = df_modified["Translation Vector (x y z)"].str.replace(r"[()]", "", regex=True).str.strip()

    # Split into 3 columns and convert to float
    df_modified[["X_pos", "Y_pos", "Z_pos"]] = df_modified["Translation Vector (x y z)"].str.split(expand=True).astype(float)
    df_modified["VectorLength"] = np.sqrt(df_modified["X_pos"] ** 2 + df_modified["Y_pos"] ** 2 + df_modified["Z_pos"] ** 2)
    df_modified = df_modified[df_modified["Z_pos"].notna()]
    df_modified["Record Time Stamp"] = pd.to_datetime(df_modified["Record Time Stamp"])
    df_modified = df_modified.sort_values("Record Time Stamp")
    df_modified["t_seconds"] = (df_modified["Record Time Stamp"] - df_modified["Record Time Stamp"].iloc[0]).dt.total_seconds()
    df_modified["Z_pos_corr"] = df_modified["Z_pos"] - np.mean(df_modified["Z_pos"].values)
    df_modified = df_modified[df_modified["t_seconds"] < (np.max(df_modified["t_seconds"].values) - 10)]

    # --- Predicted Vector (x y z) ---
    if "Predicted Translation" in df_modified.columns:
        # Clean the data by removing the empty values or Nan
        df_modified["Predicted Translation"] = (
            df_modified["Predicted Translation"]
            .astype(str)
            .str.replace(r"[()]", "", regex=True)
            .str.strip()
        )

        # Replace Nan by np.nan
        df_modified["Predicted Translation"] = df_modified["Predicted Translation"].replace("nan", np.nan)

        # Separates in 3 columns → pandas puts NaN automatically if values are missing
        pred_split = df_modified["Predicted Translation"].str.split(expand=True)
        df_modified["X_pred"] = pd.to_numeric(pred_split[0], errors="coerce")
        df_modified["Y_pred"] = pd.to_numeric(pred_split[1], errors="coerce")
        df_modified["Z_pred"] = pd.to_numeric(pred_split[2], errors="coerce")

    if df_modified["Z_pred"].notna().any():

        # Center the predicted Z position around zero (mean = 0)
        df_modified["Z_pred_corr"] = df_modified["Z_pred"] - np.nanmean(df_modified["Z_pred"].values)

        # Make the predicted Z position smooth using Savitzky-Golay filter
        z_pred_values = df_modified["Z_pred_corr"].ffill().bfill().values
        z_pred_smooth = savgol_filter(z_pred_values, window_length=21, polyorder=3)
        df_modified["Z_pred_smooth"] = z_pred_smooth

        # --- Gating Average Error Position (x y z) ---
    if "Gating Average Position Error (in mm)" in df_modified.columns:
        # Clean the data by removing the empty values or Nan
        df_modified["gating average error position in mm"] = (
            df_modified["Gating Average Position Error (in mm)"]
            .astype(str)
            .str.replace(r"[()]", "", regex=True)
            .str.strip()
        )

        # Replace Nan by np.nan
        df_modified["gating average error position in mm"] = (
            df_modified["gating average error position in mm"].replace("nan", np.nan)
        )

        # separates in 3 columns → pandas puts NaN automatically if values are missing
        err_split = df_modified["gating average error position in mm"].str.split(expand=True)
        df_modified["X_error"] = pd.to_numeric(err_split[0], errors="coerce")
        df_modified["Y_error"] = pd.to_numeric(err_split[1], errors="coerce")
        df_modified["Z_error"] = pd.to_numeric(err_split[2], errors="coerce")


    # Plot the graph of the position of the tumor in the z-direction over time

    #define parameters
    t = df_modified["t_seconds"].values  # Temps
    z_corr = df_modified["Z_pos_corr"].values  # Position corrigée
    x_corr = df_modified["X_pos"].values - np.mean(df_modified["X_pos"].values)
    y_corr = df_modified["Y_pos"].values - np.mean(df_modified["Y_pos"].values)

    #making signal smooth
    z_smooth = savgol_filter(z_corr, window_length=21, polyorder=3)
    x_smooth = savgol_filter(x_corr, window_length=21, polyorder=3)
    y_smooth = savgol_filter(y_corr, window_length=21, polyorder=3)


    #detect peaks and troughs
    peaks, _ = find_peaks(z_smooth, distance=20, prominence=0.2, width=5)
    troughs, _ = find_peaks(-z_smooth, distance=20, prominence=0.2, width=5)

    # Combine peaks and troughs to get cycle indices (Not used But was used to verify some errors)
    cycle_indices = sorted(list(peaks) + list(troughs))

    # Filter the peaks and troughs to keep only those within the range 100 to 200
    peaks_in_range = [p for p in peaks if 100 <= p < 200]
    troughs_in_range = [t for t in troughs if 100 <= t < 200]

    t_start =1
    t_end = 1500
    plt.figure(figsize=(12, 8))
    # plt.plot(t, x_corr, label="X original", alpha=0.5, color='red')
    # plt.plot(t, x_smooth, label="X smooth", color='darkred')
    # plt.plot(t[t_start:t_end], y_corr[t_start:t_end], label="Y original", alpha=0.5, color='green')
    # plt.plot(t[t_start:t_end], y_smooth[t_start:t_end], label="Y smooth", color='darkgreen')

    plt.plot(t, z_corr, label="Z original", alpha=0.5, color='blue')
    plt.plot(t, z_smooth, label="Z smooth", color='darkblue')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Z position (mm)")
    plt.legend()
    plt.title("3D Position of the Tumor Over Time (Smoothed and raw Signals)")
    plt.grid(True)
    plt.show()

    #add smoothed signal in excel
    df_modified["Z_smooth"] = z_smooth
    df_modified["X_smooth"] = x_smooth
    df_modified["Y_smooth"] = y_smooth

    # Save the modified DataFrame to a new CSV file
    df_modified.to_csv(output_path, index=False)





# Path to the original CSV file
input_csv_path = "AuditLog_syntheticMotion.csv"

# Path to save the modified CSV file
output_csv_path = "AuditLog_syntheticMotionModif.csv"

# Create a tuple with the input and output paths
Logfiles = (input_csv_path, output_csv_path)

# Call the function to create a copy and modify it partially
regularisation(Logfiles)

