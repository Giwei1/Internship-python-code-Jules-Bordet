# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 10:21:37 2025

@author: akosg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy import stats
import io

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
    raw_df.columns=["Silos","Tick","Gate","Beam","Is", "delete"]
    raw_df = raw_df[["Silos","Tick","Gate","Beam","Is"]]
    # raw_df["Silos"] = raw_df.index
    # raw_df.reset_index(inplace=True)
    
    return overall_df, silos_df, raw_df


def process_log(path, rpm):

    overall_df, silos_df, raw_df = import_multi_section_csv(path)
    
    return {
        "summary": overall_df,
        "silos": silos_df,
        "raw_data": raw_df,
        "rpm": rpm
    }

#def plot_beam_latency_from_results(results, bins=30):
    # num_files = len(results)
    # fig, axes = plt.subplots(2, num_files, figsize=(5 * num_files, 6), sharex=True, sharey=True)
    #
    # if num_files == 1:
    #     axes = [axes]  # handle single case consistently
    #
    # for i, result in enumerate(results):
    #     beam_on = result["silos"]["Beam On (ms)"].fillna(0)
    #     beam_off = result["silos"]["Beam Off (ms)"].fillna(0)
    #     num_silos = len(result["silos"])
    #     rpm = result["rpm"]
    #
    #     ax_on = axes[0, i]
    #     ax_off = axes[1, i]
    #
    #     if i ==0:
    #         ax_on.set_ylabel ("Beam On",fontsize=14)
    #         ax_off.set_ylabel ("Beam Off", fontsize=14)
    #
    #     # Beam On
    #     ax_on.hist(beam_on, bins=bins, edgecolor='black', alpha=0.7)
    #     ax_on.axvline(-0.2, linestyle=':', color='red')
    #     ax_on.axvline(0.2, linestyle=':', color='red')
    #     #ax_on.set_title(f"RPM: {rpm} | Silos: {num_silos}",fontsize=14)
    #     ax_on.grid(True, linestyle='--', alpha=0.5)
    #
    #     xlim = ax_on.get_xlim()
    #     ylim = ax_on.get_ylim()
    #     x_sym_limit = round(max(abs(ax_on.get_xlim()[0]),abs(ax_on.get_xlim()[1])),1)
    #
    #     ax_on.set_xlim([-x_sym_limit, x_sym_limit])
    #
    #
    #     count_on = len(beam_on)
    #     mean_on = np.mean(beam_on)
    #     within_range_on = np.sum((beam_on >= -0.2) & (beam_on <= 0.2))
    #     percent_within_on = 100 * within_range_on / count_on
    #     std_on = np.std(beam_on)
    #
    #     conf_interval_on = stats.t.interval(
    #         0.95, count_on - 1, loc=mean_on, scale=std_on / np.sqrt(count_on)
    #         )
    #     ci_low_on, ci_high_on = conf_interval_on
    #     stats_text_on = (f"Mean: {mean_on*1000.0:.0f} ms\n"
    #                      f"Std: {std_on*1000.0:.0f} ms\n"
    #                      # f"95% CI: [{ci_low_on:.3f}, {ci_high_on:.3f}]\n"
    #                      f"<200ms: {percent_within_on:.1f}%\n")
    #     ax_on.text(0.95, 0.8, stats_text_on, transform=ax_on.transAxes,
    #                fontsize=10, verticalalignment='top', horizontalalignment='right',
    #                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    #
    #
    #     # Beam Off
    #     ax_off.hist(beam_off, bins=bins, edgecolor='black', alpha=0.7)
    #     ax_off.axvline(-0.2, linestyle=':', color='red')
    #     ax_off.axvline(0.2, linestyle=':', color='red')
    #     # ax_off.set_title(f"Beam Off @ RPM: {rpm} | Silos: {num_silos}")
    #     ax_off.grid(True, linestyle='--', alpha=0.5)
    #     ax_off.set_xlabel("Time (s)")
    #
    #     count_off = len(beam_off)
    #     mean_off = np.mean(beam_off)
    #     within_range_off = np.sum((beam_off >= -0.2) & (beam_off <= 0.2))
    #     percent_within_off = 100 * within_range_off / count_off
    #     std_off = np.std(beam_off)
    #
    #
    #     stats_text_on = (f"Mean: {mean_off*1000:.0f} ms\n"
    #                      f"Std: {std_off*1000:.0f} ms\n"
    #                      # f"95% CI: [{ci_low_off:.3f}, {ci_high_off:.3f}]\n"
    #                      f"<200ms: {percent_within_off:.0f}%")
    #     ax_off.text(0.95, 0.8, stats_text_on, transform=ax_off.transAxes,
    #                fontsize=10, verticalalignment='top', horizontalalignment='right',
    #                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    #
    #
    # ylim = ax_on.get_ylim()
    # # ax_on.set_ylim(0,20)
    # for i, ax in enumerate(axes[1, :]):
    #     ax.text(xlim[0]+0.05*(xlim[1]-xlim[0]), ylim[1]*0.95, "late", ha='left', va='top', fontsize=9, weight='bold', color='green',bbox = dict(boxstyle='round', facecolor='lightgrey', alpha=0.8))
    #     ax.text(xlim[1]-0.05*(xlim[1]-xlim[0]), ylim[1]*0.95, "early", ha='right', va='top', fontsize=9, color='red',weight='bold',bbox = dict(boxstyle='round', facecolor='lightgrey', alpha=0.8))
    #
    # for i, ax in enumerate(axes[0, :]):
    #     ax.text(xlim[0]+0.05*(xlim[1]-xlim[0]), ylim[1]*0.95, "early", ha='left', va='top', fontsize=9, color='red',weight='bold',bbox = dict(boxstyle='round', facecolor='lightgrey', alpha=0.8))
    #     ax.text(xlim[1]-0.05*(xlim[1]-xlim[0]), ylim[1]*0.95, "late", ha='right', va='top', fontsize=9, color='green',weight='bold',bbox = dict(boxstyle='round', facecolor='lightgrey', alpha=0.8))
    #
    #
    #
    # fig.suptitle("Beam latency", fontsize=16, weight='bold')
    # # plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    # plt.tight_layout()
    # fig.savefig("Beam latency.png", dpi=600, bbox_inches='tight')
    # plt.show()
    #
    #

def plot_beam_latency_combined(results, bins=30):
    fig, (ax_on, ax_off) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    # Listes pour concaténer tous les résultats
    all_beam_on = []
    all_beam_off = []

    for result in results:
        beam_on = result["silos"]["Beam On (ms)"].fillna(0).to_numpy()
        beam_off = result["silos"]["Beam Off (ms)"].fillna(0).to_numpy()
        all_beam_on.extend(beam_on)
        all_beam_off.extend(beam_off)

    all_beam_on = np.array(all_beam_on)
    all_beam_off = np.array(all_beam_off)

    # --- BEAM ON ---
    ax_on.hist(all_beam_on, bins=bins, edgecolor='black', alpha=0.7)
    ax_on.axvline(-0.2, linestyle=':', color='red')
    ax_on.axvline(0.2, linestyle=':', color='red')
    ax_on.set_title("Beam On Latency", fontsize=14)
    ax_on.set_ylabel("Count")
    ax_on.grid(True, linestyle='--', alpha=0.5)

    count_on = len(all_beam_on)
    mean_on = np.mean(all_beam_on)
    std_on = np.std(all_beam_on)
    within_range_on = np.sum((all_beam_on >= -0.2) & (all_beam_on <= 0.2))
    percent_within_on = 100 * within_range_on / count_on

    stats_text_on = (f"Mean: {mean_on*1000:.0f} ms\n"
                     f"Std: {std_on*1000:.0f} ms\n"
                     f"<200ms: {percent_within_on:.1f}%")
    ax_on.text(0.95, 0.8, stats_text_on, transform=ax_on.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # --- BEAM OFF ---
    ax_off.hist(all_beam_off, bins=bins, edgecolor='black', alpha=0.7)
    ax_off.axvline(-0.2, linestyle=':', color='red')
    ax_off.axvline(0.2, linestyle=':', color='red')
    ax_off.set_title("Beam Off Latency", fontsize=14)
    ax_off.set_xlabel("Latency (s)")
    ax_off.set_ylabel("Count")
    ax_off.grid(True, linestyle='--', alpha=0.5)

    count_off = len(all_beam_off)
    mean_off = np.mean(all_beam_off)
    std_off = np.std(all_beam_off)
    within_range_off = np.sum((all_beam_off >= -0.2) & (all_beam_off <= 0.2))
    percent_within_off = 100 * within_range_off / count_off

    stats_text_off = (f"Mean: {mean_off*1000:.0f} ms\n"
                      f"Std: {std_off*1000:.0f} ms\n"
                      f"<200ms: {percent_within_off:.1f}%")
    ax_off.text(0.95, 0.8, stats_text_off, transform=ax_off.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    fig.suptitle("Beam Latency (Combined)", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()


logfiles = [
    ("Gating_syntheticMotion.csv", 1.0),
    # ("Logfiles/20250505_10RPM.csv", 10.0),
    ("Gating_p1_0704.csv", 10.0)
]
amplitude = 10.0  # known amplitude of sine wave (mm)

# === PROCESS BOTH FILES ===
results = [process_log(path, rpm) for path, rpm in logfiles]






plot_beam_latency_combined(results)




# # fig, axs = plt.subplots(1, 2, figsize=(14, 5))
# # for i, result in enumerate(results):
# #     axs[i].hist(result["abs_disagreement"], bins=50, edgecolor='black', alpha=0.7)
# #     axs[i].axvline(result["conf_95"], color='red', linestyle='--', label=f'95% CI = {result["conf_95"]:.3f}')
# #     axs[i].set_title(f'APM Accuracy @ {result["rpm"]:.0f} RPM')
# #     axs[i].set_xlabel('Distance to agreement (mm)')
# #     axs[i].set_ylabel('Frequency')
# #     axs[i].legend()
# #     axs[i].grid(True)

# #     # Text box
# #     textstr = '\n'.join((
# #         f'Amplitude ±{amplitude:.1f} mm',
# #         f'Cycle time: {result["cycle_time"]:.2f} s',
# #         f'Total points: {result["total_points"]}',
# #         f'Total cycles: {result["total_cycles"]:.0f}',
# #         f'95% CI: {result["conf_95"]:.3f} mm',
# #     ))
# #     props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
# #     axs[i].text(0.95, 0.95, textstr, transform=axs[i].transAxes,
# #                 fontsize=11, verticalalignment='top',
# #                 horizontalalignment='right', bbox=props)

# # plt.tight_layout()
# # plt.show()


# def plot_beam_on_off(silos_df):
#     # Extract data
#     beam_on = silos_df["Beam On (ms)"]
#     beam_off = silos_df["Beam Off (ms)"]
#     silos = silos_df["Silos"] if "Silos" in silos_df.columns else silos_df.index

#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

#     # Plot Beam On
#     colors_on = ['green' if abs(v) < 0.2 else 'red' for v in beam_on]
#     labels_on = ['Early' if v < 0 else 'Late' for v in beam_on]
#     ax1.bar(silos, beam_on, color=colors_on)
#     ax1.axhline(-0.2, linestyle=':', color='black')
#     ax1.axhline(0.2, linestyle=':', color='black')
#     ax1.set_title("Beam On (ms)")
#     ax1.set_ylabel("Time (ms)")
#     ax1.grid(True, linestyle='--', alpha=0.5)

#     # Plot Beam Off
#     colors_off = ['green' if abs(v) < 0.2 else 'red' for v in beam_off]
#     labels_off = ['Late' if v < 0 else 'Early' for v in beam_off]
#     ax2.bar(silos, beam_off, color=colors_off)
#     ax2.axhline(-0.2, linestyle=':', color='black')
#     ax2.axhline(0.2, linestyle=':', color='black')
#     ax2.set_title("Beam Off (ms)")
#     ax2.set_ylabel("Time (ms)")
#     ax2.set_xlabel("Silos")
#     ax2.grid(True, linestyle='--', alpha=0.5)

#     plt.tight_layout()
#     plt.show()



# def plot_beam_histogram(silos_df,rpm, bins=30):
#     beam_on = silos_df["Beam On (ms)"]
#     beam_off = silos_df["Beam Off (ms)"]
#     num_silos = len(silos_df)
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True, sharey=True)

#     # Beam On Histogram
#     ax1.hist(beam_on, bins=bins,edgecolor='black', alpha=0.7)
#     ax1.axvline(-0.2, linestyle=':', color='black')
#     ax1.axvline(0.2, linestyle=':', color='black')
#     ax1.set_title("Beam On")
#     ax1.set_ylabel("Number of Silos")
#     ax1.grid(True, linestyle='--', alpha=0.5)
#     # Add corner labels to Beam On
#     xlim = ax1.get_xlim()
#     ylim = ax1.get_ylim()
#     ax1.text(xlim[0]+0.05*(xlim[1]-xlim[0]), ylim[1]*0.95, "Too early", ha='left', va='top', fontsize=10, color='red')
#     ax1.text(xlim[1]-0.05*(xlim[1]-xlim[0]), ylim[1]*0.95, "Too late", ha='right', va='top', fontsize=10, color='green')



#     # Beam Off Histogram
#     ax2.hist(beam_off, bins=bins, color="blue",edgecolor='black', alpha=0.7)
#     ax2.axvline(-0.2, linestyle=':', color='black')
#     ax2.axvline(0.2, linestyle=':', color='black')
#     ax2.set_title("Beam Off")
#     ax2.set_xlabel("Time (ms)")
#     ax2.set_ylabel("Number of Silos")
#     ax2.grid(True, linestyle='--', alpha=0.5)

#     ax2.text(xlim[0]+0.05*(xlim[1]-xlim[0]), ylim[1]*0.95, "Too late", ha='left', va='top', fontsize=10, color='green')
#     ax2.text(xlim[1]-0.05*(xlim[1]-xlim[0]), ylim[1]*0.95, "Too early", ha='right', va='top', fontsize=10, color='red')

#     fig.suptitle(f"Beam latency (RPM: {rpm})", fontsize=14, weight='bold')

#     fig.text(0.99, 0.01, f"Number of silos plotted: {num_silos}",
#          ha='right', va='bottom', fontsize=10,
#          bbox=dict(facecolor='lightgrey', edgecolor='grey', boxstyle='round,pad=0.4'))


#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    

#     plt.tight_layout()
#     plt.show()


# silos_df = results[0]["silos"]

# silos_df = silos_df[(silos_df['System Excluded']==False)]
# silos_df = silos_df[(silos_df['User Excluded']==False)]
# plot_beam_on_off(silos_df)
# plot_beam_histogram(silos_df, 10.0)



# fig, axs = plt.subplots(2, 2, figsize=(10, 10),sharex=True, sharey=True)
# for i, result in enumerate(results):

#     # beam_on = silos_df["Beam On (ms)"]
#     # beam_off = silos_df["Beam Off (ms)"]
#     # num_silos = len(silos_df)
#     # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True, sharey=True)

#     # # Beam On Histogram
#     # ax1.hist(beam_on, bins=bins,edgecolor='black', alpha=0.7)
#     # ax1.axvline(-0.2, linestyle=':', color='black')
#     # ax1.axvline(0.2, linestyle=':', color='black')
#     # ax1.set_title("Beam On")
#     # ax1.set_ylabel("Number of Silos")
#     # ax1.grid(True, linestyle='--', alpha=0.5)
#     # # Add corner labels to Beam On
#     # xlim = ax1.get_xlim()
#     # ylim = ax1.get_ylim()
#     # ax1.text(xlim[0]+0.05*(xlim[1]-xlim[0]), ylim[1]*0.95, "Too early", ha='left', va='top', fontsize=10, color='red')
#     # ax1.text(xlim[1]-0.05*(xlim[1]-xlim[0]), ylim[1]*0.95, "Too late", ha='right', va='top', fontsize=10, color='green')



#     # # Beam Off Histogram
#     # ax2.hist(beam_off, bins=bins, color="blue",edgecolor='black', alpha=0.7)
#     # ax2.axvline(-0.2, linestyle=':', color='black')
#     # ax2.axvline(0.2, linestyle=':', color='black')
#     # ax2.set_title("Beam Off")
#     # ax2.set_xlabel("Time (ms)")
#     # ax2.set_ylabel("Number of Silos")
#     # ax2.grid(True, linestyle='--', alpha=0.5)

#     # ax2.text(xlim[0]+0.05*(xlim[1]-xlim[0]), ylim[1]*0.95, "Too late", ha='left', va='top', fontsize=10, color='green')
#     # ax2.text(xlim[1]-0.05*(xlim[1]-xlim[0]), ylim[1]*0.95, "Too early", ha='right', va='top', fontsize=10, color='red')

#     # fig.suptitle(f"Beam latency (RPM: {rpm})", fontsize=14, weight='bold')

#     # fig.text(0.99, 0.01, f"Number of silos plotted: {num_silos}",
#     #      ha='right', va='bottom', fontsize=10,
#     #      bbox=dict(facecolor='lightgrey', edgecolor='grey', boxstyle='round,pad=0.4'))


#     # plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    

#     # plt.tight_layout()
#     # plt.show()
    
    
#     axs[i].hist(result["abs_disagreement"], bins=50, edgecolor='black', alpha=0.7)
#     axs[i].axvline(result["conf_95"], color='red', linestyle='--', label=f'95% CI = {result["conf_95"]:.3f}')
#     axs[i].set_title(f'@ {result["rpm"]:.0f} RPM')
#     axs[i].set_xlabel('Distance to agreement (mm)')
#     axs[i].set_ylabel('Frequency')
#     axs[i].legend()
#     axs[i].grid(True)

#     # Text box
#     textstr = '\n'.join((
#         f'Amplitude ±{amplitude:.1f} mm',
#         f'Cycle time: {result["cycle_time"]:.2f} s',
#         f'Total points: {result["total_points"]}',
#         f'Total cycles: {result["total_cycles"]:.0f}',
#         f'95% CI: {result["conf_95"]:.3f} mm',
#     ))
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
#     axs[i].text(0.95, 0.95, textstr, transform=axs[i].transAxes,
#                 fontsize=11, verticalalignment='top',
#                 horizontalalignment='right', bbox=props)
# fig.suptitle(f"APM Accuracy", fontsize=14, weight='bold')
# plt.tight_layout()
# plt.show()
