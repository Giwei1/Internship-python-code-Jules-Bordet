import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def analyze_gating_error(file_path):
    """
        Imports a CSV file containing X_error, Y_error, Z_error columns, computes the error norm,
        and visualizes the results.

        The function performs the following:
            1. Computes the Euclidean norm of the 3D errors.
            2. Plots the error curves for X, Y, Z, and the norm, expressed as a percentage of the mean peak amplitude.
            3. Generates boxplots of X, Y, Z errors in % of the corresponding mean peak values.
            4. Prints basic statistics: mean and median errors in mm and %.

        Parameters:
            file_path (str): Path to the CSV file containing the required columns:
                             "X_error", "Y_error", "Z_error", and "X_smooth", "Y_smooth", "Z_smooth"
                             for normalization.

        Returns:
            None

        Notes:
            - The error curves are normalized to the mean peak value of the corresponding axis.
            - The norm is normalized using the mean of the three axes.
            - Histograms are plotted as percentage of points per error bin.
            - Raises ValueError if required columns are missing in the CSV file.
        """
    # --- Import CSV ---
    df = pd.read_csv(file_path)

    # verify columns
    for col in ["X_error", "Y_error", "Z_error"]:
        if col not in df.columns:
            raise ValueError(f"Colonne {col} manquante dans le fichier CSV")

    # Verify columns of position
    for col in ["X_smooth", "Y_smooth", "Z_smooth"]:
        if col not in df.columns:
            raise ValueError(f"Colonne {col} manquante dans le fichier CSV (nécessaire pour normaliser)")

    # --- Calcul of error norm ---
    df["ErrorNorm"] = np.sqrt(df["X_error"]**2 + df["Y_error"]**2 + df["Z_error"]**2)

    mean_peaks = {}
    for axis in ["X_smooth", "Y_smooth", "Z_smooth"]:
        values = df[axis].dropna().values
        peaks, _ = find_peaks(values)
        mean_peaks[axis] = np.mean(values[peaks]) if len(peaks) > 0 else np.mean(values)
    global_mean_peak = np.nanmean(list(mean_peaks.values()))

    # --- Data preparation ---
    signals = {
        "X_error": df["X_error"].dropna().values,
        "Y_error": df["Y_error"].dropna().values,
        "Z_error": df["Z_error"].dropna().values,
        "Norme": df["ErrorNorm"].dropna().values,
    }

    plt.figure(figsize=(10, 6))

    for label, values in signals.items():
        if len(values) == 0:
            continue

        # Normalisation with corresponding mean peak
        if label.startswith("X"):
            mean_peak = mean_peaks["X_smooth"]
        elif label.startswith("Y"):
            mean_peak = mean_peaks["Y_smooth"]
        elif label.startswith("Z"):
            mean_peak = mean_peaks["Z_smooth"]
        else:  # Norm → Mean of the 3 means
            mean_peak = global_mean_peak


        # Histogram transformed in curve
        bins = np.linspace(0, np.max(values), 50)
        counts, edges = np.histogram(values, bins=bins)
        percent = counts / counts.sum() * 100
        bin_centers = (edges[:-1] + edges[1:]) / 2

        plt.plot(bin_centers, percent, label=label)

        # ---Formating ---
    plt.xlabel("Error (mm)")
    plt.ylabel("% of points")
    plt.title("error distribution between predicted and actual position")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Boxplot in % ---
    X_err_percent = (df["X_error"].dropna() / mean_peaks["X_smooth"]) * 100
    Y_err_percent = (df["Y_error"].dropna() / mean_peaks["Y_smooth"]) * 100
    Z_err_percent = (df["Z_error"].dropna() / mean_peaks["Z_smooth"]) * 100

    plt.figure(figsize=(8, 6))
    plt.boxplot([X_err_percent, Y_err_percent, Z_err_percent],
                labels=["X_error", "Y_error", "Z_error"])
    plt.ylabel("Error (% of mean peak)")
    plt.title("Boxplot of prediction errors (in %)")
    plt.grid(True)
    plt.show()

    # --- Principal statistics ---
    print("📊 Principal statistics (mm et %) :")
    for label, values in signals.items():
        if len(values) > 0:
            if label.startswith("X"):
                ref = mean_peaks["X_smooth"]
            elif label.startswith("Y"):
                ref = mean_peaks["Y_smooth"]
            elif label.startswith("Z"):
                ref = mean_peaks["Z_smooth"]
            else:
                ref = global_mean_peak
            mean_mm = np.mean(values)
            mean_percent = (mean_mm / ref) * 100
            print(f"{label} → Mean = {mean_mm:.2f} mm ({mean_percent:.1f}%) | Median = {np.median(values):.2f} mm")


def main():

    file_path = "AuditLog_p1_test_200Z_0801_11_30Modified.csv"  # <-- à adapter
    print(f"📂 Importation of file : {file_path}")
    analyze_gating_error(file_path)

if __name__ == "__main__":
    main()
