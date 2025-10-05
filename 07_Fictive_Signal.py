import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_synthetic_motion(output_csv, step=0.1, total_time=300):
    """
    Generates an artificial Motion Controller signal:
    - 0–100 s: sine wave with variable offset
    - 100–200 s: sine wave with variable amplitude
    - 200–300 s: combination of variable offset and amplitude
    """
    #Time vector
    t = np.arange(0, total_time, step)

    # Base signal (sine wave)
    base_freq = 0.2  # Hz
    base_signal = 2*np.sin(2 * np.pi * base_freq * t)

    # Offsets variables
    offset1 = 1.5 * np.sin(2 * np.pi * 0.02 * t)   # offset sinus lent
    offset2 = 2 * np.sin(2 * np.pi * 0.02* t)  # amplitude ±8, fréquence 0.2 Hz

    # Amplitudes variables
    amplitude1 = 1 + 2 * np.sin( np.pi * 0.01 * t)
    amplitude2 = np.linspace(0.5, 2, len(t))              # amplitude en rampe

    # Signal construction by segments
    signal = np.zeros_like(t)

    # Phase 1 (0–100s) : sinus + offset
    mask1 = (t < 100)
    signal[mask1] = base_signal[mask1] + offset1[mask1]

    # Phase 2 (100–200s) : sinus with amplitude changing
    mask2 = (t >= 100) & (t < 200)
    signal[mask2] = amplitude1[mask2] * base_signal[mask2]

    # Phase 3 (200–300s) : sinus with offset + amplitude
    mask3 = (t >= 200)
    signal[mask3] = amplitude1[mask3] * base_signal[mask3] + offset2[mask3]

    # Creation of DataFrame in Motion Controller format
    df = pd.DataFrame({
        "Time": t,
        "Inf/Sup": signal,     # axe Z
        "AP": np.zeros_like(t),  # tu peux mettre d’autres signaux si tu veux
        "LR": np.zeros_like(t)
    })

    df.to_csv(output_csv, index=False)
    print(f"Generated file : {output_csv}")

    # Vérification visuelle
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, label="Synthetic signal (Inf/Sup)")
    plt.axvline(100, color='red', linestyle="--", label="Phase changement")
    plt.axvline(200, color='red', linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    return df


# Usage
generate_synthetic_motion("synthetic_motion.csv", step=0.1, total_time=300)
