import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def time_regularisation(path):
    """
    This function processes a CSV file to regularize time and 3D position data:
    1. Reads the input CSV file and removes existing regulated columns if present.
    2. Extracts irregular time (`t_seconds`) and smoothed 3D position data (`X_smooth`, `Y_smooth`, `Z_smooth`).
    3. Creates a regular time vector (`t_regular`) with a fixed step size.
    4. Performs linear interpolation of the 3D position data to align with the regular time vector.
    5. Optionally plots a comparison of smoothed and interpolated data for a specific time range.
    6. Identifies the start and end times of specific beam states (`HoldBeam` or `DontHoldBeam`) and calculates their duration.
    7. Appends the interpolated data as new columns (`t_regulated`, `X_regulated`, `Y_regulated`, `Z_regulated`) to the DataFrame.
    8. Saves the updated DataFrame back to the same CSV file.
    """
    df = pd.read_csv(path)

    # Verify that the columns exist
    if "t_regulated" in df.columns:
        df = df.drop(columns=["t_regulated"])
    if "Z_regulated" in df.columns:
        df = df.drop(columns=["Z_regulated"])
    if "X_regulated" in df.columns:
        df = df.drop(columns=["X_regulated"])
    if "Y_regulated" in df.columns:
        df = df.drop(columns=["Y_regulated"])

    # Extracting the relevant columns
    t_irregular = df["t_seconds"].values
    z_smooth = df["Z_smooth"].values
    x_smooth = df["X_smooth"].values
    y_smooth = df["Y_smooth"].values
    beam_states = df["BEAM HOLD State"].values

    # creation of a regular time vector
    t_min = np.nanmin(t_irregular)
    t_max = np.nanmax(t_irregular)
    step = 0.1  # can choose the step size
    t_regular = np.arange(t_min, t_max, step)

    # linear interpolation of the 3D values
    z_interp = np.interp(t_regular, t_irregular, z_smooth)
    x_interp = np.interp(t_regular, t_irregular, x_smooth)
    y_interp = np.interp(t_regular, t_irregular, y_smooth)

    #plot a random part of the data to verify the regularisation (to verify but not necessary)
    #t_start = 1.0  # secondes
    #t_end = 100.0  # secondes

    #select the data in time range for irregular time data
    #mask_irregular = (t_irregular >= t_start) & (t_irregular <= t_end)
    #t_irregular_slice = t_irregular[mask_irregular]
    #z_smooth_slice = z_smooth[mask_irregular]
    #x_smooth_slice = x_smooth[mask_irregular]
    #y_smooth_slice = y_smooth[mask_irregular]

    #select the data in time range for regular time data
    #mask_regular = (t_regular >= t_start) & (t_regular <= t_end)
    #t_regular_slice = t_regular[mask_regular]
    #z_interp_slice = z_interp[mask_regular]
    #x_interp_slice = x_interp[mask_regular]
    #y_interp_slice = y_interp[mask_regular]

    #plot the data
    # Plot X, Y, Z lissés vs interpolés
    # plt.figure(figsize=(12, 8))
    # plt.plot(t_irregular_slice, x_smooth_slice, 'o', label='X_smooth', markersize=4, color='red')
    # plt.plot(t_regular_slice, x_interp_slice, 'x', label='X_interp', markersize=4, color='darkred')
    #
    # plt.plot(t_irregular_slice, y_smooth_slice, 'o', label='Y_smooth', markersize=4, color='green')
    # plt.plot(t_regular_slice, y_interp_slice, 'x', label='Y_interp', markersize=4, color='darkgreen')
    #
    # plt.plot(t_irregular_slice, z_smooth_slice, 'o', label='Z_smooth', markersize=4, color='blue')
    # plt.plot(t_regular_slice, z_interp_slice, 'x', label='Z_interp', markersize=4, color='orange')
    #
    # plt.legend()
    # plt.grid(True)
    # plt.title(f" 3D interpolation between {t_start}s and {t_end}s")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Position (cm)")
    # plt.show()

    # add to the data to document
    df_interp = pd.DataFrame({
        "t_regulated": t_regular,
        "X_regulated": x_interp,
        "Y_regulated": y_interp,
        "Z_regulated": z_interp
    })

    #Prints to verify the information
    #find the indices where the beam state is "ON"
    start_idx = None
    for i in range(len(beam_states)):
        if beam_states[i] in ['HoldBeam', 'DontHoldBeam']:
            start_idx = i
            break
    if start_idx is not None:
        start_time = t_irregular[start_idx]
        print(f"Beginning of index changing {start_idx}, time = {start_time:.2f} s")

        #find the end index where the beam is useful
        t_valid = t_irregular[~np.isnan(t_irregular)]
        if len(t_valid) == 0:
            print(" All values are Nan!")
        else:
            end_time = t_valid[-1]

            print(f"Beginning of 'HoldBeam' stateor 'DontHoldBeam' state {start_time:.2f} s")
            print(f"Last valid time : {end_time:.2f} s")

            duration = end_time - start_time
            print(f"Total actif duration : {duration:.2f} s")
    else:
        print("No 'HoldBeam' state or 'DontHoldBeam' state detected.")

    #Add the interpolated data to the original dataframe and save it
    df = pd.concat([df, df_interp], axis=1)
    df.to_csv(path, index=False)

def MotionFile(input_csv, output_csv, limited_output_csv, begin=0.0, end=None, limit=False, threshold_factor=1.5):
    """
This code processes a CSV file to limit extreme peaks in Z_regulated, scale the data, and export a motion file:
1. Reads the input CSV file and checks for the presence of required columns (`t_regulated`, `Z_regulated`).
2. Sets the end time to the maximum value in `t_regulated` if not provided.
3. If `limit` is enabled:
   - Limits extreme peaks in `Z_regulated` based on a threshold derived from the mean peak value.
   - Scales `Z_regulated` to ensure its range is 36 and centers it around zero.
   - Scales `X_regulated` and `Y_regulated` proportionally to `Z_regulated`.
   - Ensures the 3D vector norms do not exceed a maximum limit (default 20).
4. Filters the data based on the specified time range (`begin` to `end`).
5. Creates a new DataFrame with time and regulated 3D position data (`Inf/Sup`, `AP`, `LR`).
6. Saves the processed data to the specified output CSV file.
"""

    df = pd.read_csv(input_csv)

    if "t_regulated" not in df.columns or "Z_regulated" not in df.columns:
        print("Colonnes t_regulated ou Z_regulated non trouvées.")
        return

    if end is None:
        end = df["t_regulated"].max()

    if limit:
        print("Limiting extreme peaks in Z_regulated...")
        z = df["Z_regulated"].values
        x = df["X_regulated"].values
        y = df["Y_regulated"].values

        # Step 1 : limitation Z peaks
        peaks_pos, _ = find_peaks(z)
        peaks_neg, _ = find_peaks(-z)
        all_peaks = np.concatenate([peaks_pos, peaks_neg])
        peak_values = np.abs(z[all_peaks])
        mean_peak_value = np.mean(peak_values)
        threshold = threshold_factor * mean_peak_value
        print(f"Mean peak value: {mean_peak_value:.3f}, Threshold: {threshold:.3f}")
        z_limited = np.clip(z, -threshold, threshold)

        # Step 2 : ajust Z So Z_max - Z_min = 40 or 36 (depends if Z limit 20mm or 18mm
        z_min = np.min(z_limited)
        z_max = np.max(z_limited)
        z_range = z_max - z_min
        if z_range == 0:
            scale_z = 1.0
            z_center = 0.0
            print("Z range is zero. No scaling applied.")
        else:
            scale_z = 36.0 / z_range
            z_center = (z_max + z_min) / 2.0

        # centering and scaling Z
        z_scaled = (z_limited - z_center) * scale_z
        x_scaled = x * scale_z
        y_scaled = y * scale_z

        # Step 3 : ajust X and Y proportionally to Z so that max norm = 20
        NormLim = 20.0
        norms = np.sqrt(x_scaled**2 + y_scaled**2 + z_scaled**2)
        max_norm = np.max(norms)
        if max_norm > NormLim:
            scale_all = (NormLim / max_norm)
            x_scaled *= scale_all
            y_scaled *= scale_all
            z_scaled *= scale_all
            print(f"Norm max={max_norm:.3f} > 20, scaling ALL by {scale_all:.3f}")
        else:
            print(f"Norm max={max_norm:.3f} ≤ 20, no further scaling needed")

        # Mise à jour DataFrame
        df["Z_regulated"] = z_scaled
        df["X_regulated"] = x_scaled
        df["Y_regulated"] = y_scaled

    # Filter data by time range
    mask = (df["t_regulated"] >= begin) & (df["t_regulated"] <= end)
    df_slice = df.loc[mask].copy()
    df_slice["Time"] = df_slice["t_regulated"] - df_slice["t_regulated"].iloc[0]

    df_export = pd.DataFrame({
        "Time": df_slice["Time"],
        "Inf/Sup": df_slice["Z_regulated"],
        "AP": df_slice["X_regulated"],
        "LR": df_slice["Y_regulated"],
    })

    output_path = limited_output_csv if limit else output_csv
    df_export.to_csv(output_path, index=False)
    print(f"Motion file saved to {output_path}")



def plot_limited_vs_nonlimited(path, begin=0.0, end=None, threshold_factor=1.5):
    """
    This code processes and visualizes the `Z_regulated` signal from a CSV file:
    1. Reads the input CSV file and checks for the presence of required columns (`t_regulated`, `Z_regulated`).
    2. Sets the end time to the maximum value in `t_regulated` if not provided.
    3. Filters the data based on the specified time range (`begin` to `end`).
    4. Identifies peaks and troughs in the `Z_regulated` signal and calculates a threshold based on the mean peak value.
    5. Limits the `Z_regulated` signal to the calculated threshold to reduce extreme peaks.
    6. Plots the original and limited `Z_regulated` signals for comparison over the specified time range.
    """
    # Import CSV
    df = pd.read_csv(path)
    if "t_regulated" not in df.columns or "Z_regulated" not in df.columns:
        print("Colonnes manquantes dans le fichier.")
        return
    if end is None:
        end = df["t_regulated"].max()

    # Filter data by time range
    mask = (df["t_regulated"] >= begin) & (df["t_regulated"] <= end)
    t = df.loc[mask, "t_regulated"].values
    z_original = df.loc[mask, "Z_regulated"].values

    # Peaks detection and threshold calculation
    z_limited = z_original.copy()

    # Detection of peaks and maximum value
    peaks_pos, _ = find_peaks(z_limited)
    peaks_neg, _ = find_peaks(-z_limited)
    all_peaks = np.concatenate([peaks_pos, peaks_neg])
    peak_values = np.abs(z_limited[all_peaks])
    mean_peak_value = np.mean(peak_values)
    threshold = threshold_factor * mean_peak_value

    # Limitation ov Z position
    z_limited = np.clip(z_limited, -threshold, threshold)

    # Plot of signals
    plt.figure(figsize=(12, 6))
    plt.plot(t, z_original, label="Z Position (original)", color='blue')
    plt.plot(t, z_limited, label="Z Position (limited)", color='orange', linestyle='--')
    plt.title(f"Z Position: original vs limited, Time {begin}-{end}s")
    plt.xlabel("Time (s)")
    plt.ylabel("Z Position (mm)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_norms_before_after(input_csv, begin=0.0, end=None, threshold_factor=1.5):
    """
    This code processes and visualizes the 3D norms of regulated position data from a CSV file:
    1. Reads the input CSV file and sets the end time to the maximum value in `t_regulated` if not provided.
    2. Filters the data based on the specified time range (`begin` to `end`).
    3. Calculates the original 3D norms (`norms_original`) using `X_regulated`, `Y_regulated`, and `Z_regulated`.
    4. Limits extreme peaks in `Z_regulated` based on a threshold derived from the mean peak value of detected peaks.
    5. Scales `Z_regulated` to ensure its maximum value is 20, and proportionally scales `X_regulated` and `Y_regulated`.
    6. If the maximum 3D norm exceeds 20, applies additional scaling to all axes to ensure the norm is within the limit.
    7. Calculates and compares the norms before and after scaling (`norms_scaled` and `final_norms`).
    8. Plots the scaled norms and highlights the limit of 20 for comparison.
    9. Prints the maximum values of the original, scaled, and final norms.
    """
    # Import CSV
    df = pd.read_csv(input_csv)
    if end is None:
        end = df["t_regulated"].max()

    # Filter data by time range
    mask = (df["t_regulated"] >= begin) & (df["t_regulated"] <= end)
    df_slice = df.loc[mask].copy()
    t = df_slice["t_regulated"].values
    x = df_slice["X_regulated"].values
    y = df_slice["Y_regulated"].values
    z = df_slice["Z_regulated"].values

    # Original norm
    norms_original = np.sqrt(x**2 + y**2 + z**2)

    # Z limitation
    peaks_pos, _ = find_peaks(z)
    peaks_neg, _ = find_peaks(-z)
    all_peaks = np.concatenate([peaks_pos, peaks_neg])
    peak_values = np.abs(z[all_peaks])
    mean_peak_value = np.mean(peak_values)
    threshold = threshold_factor * mean_peak_value
    z_limited = np.clip(z, -threshold, threshold)

    # Scaling Z to have max 20 or 18
    z_max = np.max(np.abs(z_limited))
    scale_z = 18.0 / z_max if z_max != 0 else 1.0
    z_scaled = z_limited * scale_z
    x_scaled = x * scale_z
    y_scaled = y * scale_z

    # Norm after Z scaling
    norms_scaled = np.sqrt(x_scaled**2 + y_scaled**2 + z_scaled**2)
    max_norm_scaled = np.max(norms_scaled)
    NormLim = 20.0
    # if norm > 20, scaling X/Y
    if max_norm_scaled > NormLim:
        scale_all = NormLim / max_norm_scaled
        x_scaled *= scale_all
        y_scaled *= scale_all
        z_scaled *= scale_all
        print(f"Final scaling by {scale_all:.3f} to keep norm ≤ 25")

    final_norms = np.sqrt(x_scaled**2 + y_scaled**2 + z_scaled**2)

    # Plot of norms (before and after)
    plt.figure(figsize=(12, 6))
    plt.plot(t, norms_scaled, label="Norm after z scaling ", color='orange', linestyle='--')
    plt.plot(t, final_norms, label="Final norm after x,y rescaling", color='green', linestyle='-.')
    plt.axhline(20, color='red', linestyle=':', label="Limit = 20")
    plt.title(f"Norm comparaison after and before rescaling 3D | {begin}-{end}s")
    plt.xlabel("Time (s)")
    plt.ylabel("3D Norm (mm)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Norm max after Z scaling  : {max_norm_scaled:.3f}")
    print(f"Norm max final : {np.max(final_norms):.3f}")


#import the modified data file
path = "AuditLog_p2_1119Modified.csv"
new_file = "p2_1119_sample1_MC.csv"
limited_file = "p2_1119_sample1_MClimited.csv"

#call the time_regularisation function
time_regularisation(path)
MotionFile(path, new_file, limited_file, 115, 415, limit= True, threshold_factor=2)
plot_limited_vs_nonlimited(path, begin=115, end=415, threshold_factor=2)
plot_norms_before_after(path, begin=115, end=415, threshold_factor=2)