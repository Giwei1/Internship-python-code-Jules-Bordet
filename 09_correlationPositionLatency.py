import pandas as pd
import io
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import correlate
from scipy.signal import find_peaks


def import_multi_section_csv(filepath):
    # Read the whole file into memory as lines
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find where each section starts
    section_starts = {}
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if 'overall average' in line_lower or 'overall average/max/min' in line_lower:
            section_starts['overall'] = i
        elif 'silos' in line_lower:
            section_starts['silos'] = i
        elif 'raw data' in line_lower:
            section_starts['raw'] = i

    # Make sure all sections are found
    if not all(k in section_starts for k in ['overall', 'silos', 'raw']):
        raise ValueError("Could not find all required sections.")

    # Helper to get the lines for each section
    def extract_section(start, end):
        section_lines = lines[start + 1:end]  # skip the section title line
        section_text = ''.join(section_lines).strip()
        return pd.read_csv(io.StringIO(section_text))

    # Determine the range for each section
    overall_df = extract_section(section_starts['overall'], section_starts['silos'])
    silos_df = extract_section(section_starts['silos'], section_starts['raw'])
    raw_df = extract_section(section_starts['raw'], len(lines))
    raw_df.reset_index(inplace=True)
    raw_df.columns = ["Silos", "Tick", "Gate", "Beam", "Is", "delete"]
    raw_df = raw_df[["Silos", "Tick", "Gate", "Beam", "Is"]]
    # raw_df["Silos"] = raw_df.index
    # raw_df.reset_index(inplace=True)

    return overall_df, silos_df, raw_df


def import_auditlog(auditlog_file):
    df = pd.read_csv(auditlog_file)
    return df

# ------------------- Analyse par pic -------------------
def analyze_peaks_vs_latency(df_audit, raw_latency):
    t = df_audit["t_seconds"].values
    z = df_audit["Z_smooth"].values
    x = df_audit["X_smooth"].values
    y = df_audit["Y_smooth"].values
    norm = np.sqrt(x**2 + y**2 + z**2)

    results = []

    for silo in raw_latency["Silos"].unique():
        silo_data = raw_latency[raw_latency["Silos"] == silo]
        latency_signal = silo_data["Is"].values
        # On suppose que chaque tick correspond approximativement à un temps
        t_silo = np.linspace(0, len(latency_signal), len(latency_signal))

        # Détecter les pics Z pour ce silo
        peaks, properties = find_peaks(z[:len(latency_signal)], prominence=1)  # ajuster prominence si nécessaire
        peak_heights = z[peaks]

        # Associer la latence au pic le plus proche
        latencies = []
        for pk in peaks:
            if pk < len(latency_signal):
                latencies.append(latency_signal[pk])
            else:
                latencies.append(np.nan)

        # Stocker les résultats
        results.append(pd.DataFrame({
            "Silo": silo,
            "Peak_Amplitude": peak_heights,
            "Latency": latencies
        }))

    df_results = pd.concat(results, ignore_index=True)
    return df_results

# ------------------- Plot -------------------
def plot_latency_vs_peak(df_results):
    plt.figure(figsize=(10, 6))
    for silo in df_results["Silo"].unique():
        silo_data = df_results[df_results["Silo"] == silo]
        plt.scatter(silo_data["Peak_Amplitude"], silo_data["Latency"], label=f"Silo {silo}")

    plt.xlabel("Amplitude du pic (mm)")
    plt.ylabel("Latency (Is)")
    plt.title("Latency vs Amplitude de pic pour chaque Silo")
    plt.grid(True)
    plt.legend()
    plt.show()

# ------------------- Main -------------------
def main(auditlog_file, latency_file):
    df_audit = import_auditlog(auditlog_file)
    print(df_audit.columns)

    overall, silos, raw_latency = import_multi_section_csv(latency_file)
    df_results = analyze_peaks_vs_latency(df_audit, raw_latency)
    print(df_results.head())
    plot_latency_vs_peak(df_results)

# ------------------- Exécution -------------------
if __name__ == "__main__":
    auditlog_file = "AuditLog_p1_test_200Z_0801_11_30Modified.csv"
    latency_file = "Gating_p1_test_200Z_0801_12_00.csv"
    main(auditlog_file, latency_file)