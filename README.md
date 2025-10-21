# Internship-python-code-Jules-Bordet
Code used to compare medical data and measurement on the Elekta unity using the Zeus MRgRT Phantom during an internship of 4 weaks in Jules Bordet in radiophysics
# MRI–Phantom Motion Analysis

This repository contains a collection of Python scripts developed to analyze, process, and compare MRI and phantom motion data used in irradiation and latency experiments.  
All scripts were written by **Gilles Weinhofer**, **except for** `01_readaudit.py` and `02_latency.py`, with the latter being **slightly modified** to include the `gating.csv` file functionality.

---

## 📂 Files Overview

### `01_readaudit.py`
Original external file — not modified.  
Used as a reference in early stages of the project.

---

### `02_latency.py`
This file calculates the **beam latency** during irradiation.  
It uses the document shared by *Motion Control* after testing, named **`gating.csv`**, which contains all irradiation details.  

➡️ **How to use:**  
Insert your `gating.csv` file at **line 21** (where the CSV file is read) to visualize the latency data.  
A graph will then display all irradiation timing and latency information.

---

### `03_regularisation_audit.py`
This script processes the **`auditlog.csv`** file provided by the MRI system after measurements.  
It applies a series of necessary modifications and saves a new version of the file to avoid repeating the same preprocessing steps later.

➡️ **How to use:**
1. Provide your raw `auditlog.csv` file.  
2. Choose a name for the new processed copy — recommended: **`auditlogModified.csv`**.

➡️ **Main features:**
- Splits the *vectorPosition* column into **X, Y, Z** components.  
- Splits the *predictedPosition* column the same way.  
- Computes the **vector magnitude**.  
- Corrects **Z offset** (removes unnecessary nonzero baseline).  
- Calculates **measurement time** starting from 0 s.  
- Adds **smoothed position columns** to reduce noise.

---

### `04_adaptation_MotionControl.py`
This script takes the **modified auditlog file** (from `03_regularisation_audit.py`) and creates new files readable by the **Motion Control phantom simulator**.

➡️ **Outputs:**
1. A **normal output file**: interpolated positions with constant time steps.  
2. A **limited output file** (optional): constrained by the physical limitations of the phantom.

➡️ **How to use:**
- Set `limit=True` or `limit=False` in the `Motioncontrol()` function to generate the desired version.  
- Choose output file names for both versions.

➡️ **Adjustable parameters:**
- **Threshold**: controls how aggressively sharp Z peaks are cut (default: `max = threshold × mean peak value`).  
- **Max norm**: default = 20 ; can be changed.  
  - Changing this modifies the X/Y/Z ratio (Z remains unchanged).

---

### `05_comparatif.py`
Primarily a **testing script**, not essential for the final workflow.  
It compares the **raw auditlog** with the **modified auditlog**, to visualize how the processed data differ from the original patient data.

➡️ **Potential improvement:**  
Currently, the script only handles one auditlog file.  
It can be generalized to compare **two different auditlog files** (e.g., patient vs phantom measurements).

---

### `06_MRI_phantom_comparing.py`
The **main analysis script** and the most comprehensive one.  
It compares MRI and phantom motion data to assess the **accuracy, latency, precision, and reliability** of both systems.

➡️ **Inputs:**
1. The file used as **input to the phantom** for the test.  
2. The **modified auditlog** from the MRI **after phantom measurements**.

➡️ **Modes (set in the main function):**
1. **Mode 1:** Manual shifting — user chooses the time shift to align both signals.  
2. **Mode 2:** Automatic alignment — detects the end of phantom motion and aligns accordingly.  
3. **Mode 3:** No shifting applied.

After alignment, an optimization function fine-tunes the signal alignment by minimizing the error between both signals.  
You can define the search range (in seconds) for this time shift — estimated visually by plotting the two signals.

➡️ **Notes:**
- The script includes functions to compute multiple error metrics and compare all relevant parameters.
- The **`factor`** parameter refers to the multiplication factor applied in *Motion Control*.

---

### `07_Fictive_Signal.py`
This file generates a **synthetic (fictive) motion signal** from scratch.  
It is mostly used as a **reference or testing file** and should not be modified much.

➡️ **How to use:**  
Simply define the desired **output CSV file name** at the end of the script.  
Can be adapted to generate other fictive signals with different characteristics.

---

### `08_IrradiationPlot.py`
An **extra visualization file**, generating **graphs and statistical plots** that display prediction errors (e.g., boxplots).

➡️ **Input:**  
The **modified auditlog** file obtained after phantom measurement.

---

### `09_correlationPositionLatency.py`
An **unfinished exploratory script**, kept intentionally for future researchers.  
It investigates potential **correlations** such as:
- Prediction error vs latency  
- Prediction error vs irradiation on/off  
- And other possible relationships.

---

### `testplot.py`
A simple visualization script that displays **both signals before modification**, useful for quick visual checks.

---

## 🧠 Notes
- The focus of this repository is **motion tracking validation** between MRI and phantom systems.  
- Graphs and performance indicators are detailed in the internship report accompanying this project.  
- Some thresholds and parameters (e.g., limits, smoothing constants, shifting range) can be manually tuned depending on the dataset.

---

## 🧩 Author
**Developed by:** *Gilles Weinhofer*  
**Main contributions:** Files `03` → `09` and modifications to `02_latency.py`.

---

