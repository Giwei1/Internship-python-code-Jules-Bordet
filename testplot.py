import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
df1 = pd.read_csv('p2_1119_sample1_MClimited.csv')
df2 = pd.read_csv('AuditLog_p2_Fx2Modified.csv')

# Extraire les colonnes nécessaires
t1 = df1["Time"].values
z_position = df1["Inf/Sup"].values * 1.5  # Conversion si nécessaire

t2 = df2["t_seconds"].values
z_audit = df2["Z_pos"].values  # Remplacez par la colonne appropriée

# Tracer les graphes
plt.figure(figsize=(10, 12))

# Premier graphe
plt.subplot(2, 1, 1)
plt.plot(t1, z_position, label="Position Z", color='blue', linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("Position Z (mm)")
plt.title("Position Z in time (Phantom data)")
plt.grid(True)
plt.legend()

# Deuxième graphe
plt.subplot(2, 1, 2)
plt.plot(t2, z_audit, label="Audit Log Z", color='red', linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("Position Z (mm)")
plt.title(" Z position in time (IRM data)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# use of this to choose  a time range to zoom in to import on motion control
# Charger les données
df = pd.read_csv('AuditLog_p2_1119Modified.csv')

# Extraire les colonnes nécessaires
t = df["t_seconds"].values
z_pos = df["Z_pos"].values

# Tracer le graphe complet
plt.figure(figsize=(10, 6))
plt.plot(t, z_pos, label="Z Position", color='blue', linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("Z Position (mm)")
plt.title("Complete Z Position Over Time")
plt.grid(True)
plt.legend()
plt.show()