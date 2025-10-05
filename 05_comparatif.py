import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#compare the given data to the phantom with the raw patient Data (before any modification)
# This file is not important since it just compare 2 columns that differences by a modification chosen by us
# Load the modified audit log CSV file
df = pd.read_csv("AuditLog_0630Modified.csv")

z_raw = df["Z_pos_corr"].values
z_smooth = df["Z_smooth"].values
t_start = 100.0
t_end = 110.0

# Check if the lengths of the two columns are the same
assert len(z_raw) == len(z_smooth), "Les colonnes n'ont pas la même taille !"
#calculate the error between the two columns
errors = z_raw - z_smooth
errors = np.where(np.isnan(errors), 0, errors) #replace Nan values with 0
rmse = np.sqrt(np.mean(errors**2)) #average error
max_error = np.max(np.abs(errors)) # maximum error
sum_error = np.sum(errors)                # total error
sum_abs_error = np.sum(np.abs(errors))   # total absolute error


print(f"RMSE (Quadratic mean error) : {rmse:.4f} mm")
print(f"Maximum Error : {max_error:.4f} mm")
print(f"Error addition : {sum_error:.4f} mm")
print(f"Absolute error addition : {sum_abs_error:.4f} mm")


#Plot the data (error)
t = df["t_seconds"].values
mask = (t >= t_start) & (t <= t_end)
t_zoom = t[mask]
errors_zoom = errors[mask]
z_zoom = z_raw[mask]
z_smooth_zoom = z_smooth[mask]

plt.figure(figsize=(10, 6))
plt.plot(t_zoom, errors_zoom, label="Erreur (Z_brut - Z_smooth)", color='red')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("Error Z (cm)")
plt.title("Error point by point between Z brut and Z lissé")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t_zoom, z_zoom, label="Z raw", alpha=0.6)
plt.plot(t_zoom, z_smooth_zoom, label="Z smooth", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Position Z (cm)")
plt.title("Comparaison Z raw vs Z smooth")
plt.legend()
plt.grid(True)
plt.show()