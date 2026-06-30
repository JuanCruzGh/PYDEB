# -*- coding: utf-8 -*-


#%% Daily Melt Evolution

"""
pydeb_plots.py

Plot daily melt statistics from DEB-MODEL outputs.

Features
--------
- Daily mean debris melt
- Daily mean snow melt
- Daily min/max transparent envelopes
- Cumulative debris melt
- Cumulative snow melt

Input
-----
e.g.: outputs_4500_stake_hs_2023.csv
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =============================================================================
# LOAD MODEL OUTPUTS
# =============================================================================
df = pd.read_csv('outputs_4500_stake_hs_2023.csv')

# Parse dates
df['Fecha'] = pd.to_datetime(df['Fecha'])

# =============================================================================
# DAILY STATISTICS
# =============================================================================
daily = (
    df
    .set_index('Fecha')
    .resample('D')
    .agg({
        'DebrisMelt_m_we': ['mean', 'min', 'max', 'sum'],
        'SnowMelt_m_we':  ['mean', 'min', 'max', 'sum']
    })
)

# Flatten multi-index columns
daily.columns = [
    'deb_mean', 'deb_min', 'deb_max', 'deb_sum',
    'snow_mean', 'snow_min', 'snow_max', 'snow_sum'
]

# =============================================================================
# CUMULATIVE MELT
# =============================================================================
daily['deb_cum'] = daily['deb_sum'].cumsum()
daily['snow_cum'] = daily['snow_sum'].cumsum()

# =============================================================================
# PLOT
# =============================================================================
fig, ax1 = plt.subplots(figsize=(10, 5))

# -------------------------------------------------------------------------
# DAILY MEAN MELT
# -------------------------------------------------------------------------
ax1.plot(
    daily.index,
    daily['deb_mean'],
    linewidth=1.8,
    label='Daily debris melt'
)

ax1.plot(
    daily.index,
    daily['snow_mean'],
    linewidth=1.8,
    label='Daily snow melt'
)

# -------------------------------------------------------------------------
# DAILY MIN/MAX ENVELOPES
# -------------------------------------------------------------------------
ax1.fill_between(
    daily.index,
    daily['deb_min'],
    daily['deb_max'],
    alpha=0.20,
    label='Debris melt range'
)

ax1.fill_between(
    daily.index,
    daily['snow_min'],
    daily['snow_max'],
    alpha=0.20,
    label='Snow melt range'
)

# Left axis formatting
ax1.set_ylabel('Daily melt (m w.e. day$^{-1}$)')
ax1.set_xlabel('Fecha')

# =============================================================================
# SECONDARY AXIS: CUMULATIVE MELT
# =============================================================================
ax2 = ax1.twinx()

ax2.plot(
    daily.index,
    daily['deb_cum'],
    linewidth=2.2,
    linestyle='--',
    label='Cumulative debris melt'
)

ax2.plot(
    daily.index,
    daily['snow_cum'],
    linewidth=2.2,
    linestyle='--',
    label='Cumulative snow melt'
)

ax2.set_ylabel('Cumulative melt (m w.e.)')

# =============================================================================
# FORMATTING
# =============================================================================
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

fig.autofmt_xdate()

plt.title('Horcones Superior - Daily Melt Evolution')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc='upper left',
    frameon=False
)

plt.grid(False)

plt.tight_layout()
plt.show()

#%% 
# =============================================================================
# PLOT: ENERGY PARTITIONING THROUGH TIME
# =============================================================================

rho_w = 999.7
L_f   = 334000

# =============================================================================
# PLOT: ENERGY PARTITIONING THROUGH TIME
# =============================================================================

plot_dates = daily.index

# -------------------------------------------------------------------------
# DAILY ENERGY TERMS
# -------------------------------------------------------------------------

daily_energy = (
    df
    .set_index('Fecha')
    .resample('D')
    .agg({
        'TotalMelt_m_we': 'sum',
        'Sublimation_W_m2': 'mean',
        'Deposition_W_m2': 'mean'
    })
)

# Melt energy flux equivalent (W/m²)
daily_energy['MeltEnergy_W_m2'] = (
    daily_energy['TotalMelt_m_we'] * rho_w * L_f / 86400
)

# Rename for simplicity
melt_energy = daily_energy['MeltEnergy_W_m2']
subl_energy = daily_energy['Sublimation_W_m2']
deposition_energy = daily_energy['Deposition_W_m2']

# Total exchanged energy
total_energy = (
    np.abs(melt_energy) +
    np.abs(subl_energy) +
    np.abs(deposition_energy)
)

total_energy[total_energy == 0] = np.nan

# -------------------------------------------------------------------------
# RELATIVE CONTRIBUTIONS (%)
# -------------------------------------------------------------------------
melt_pct = 100 * np.abs(melt_energy) / total_energy
subl_pct = 100 * np.abs(subl_energy) / total_energy
deposition_pct = 100 * np.abs(deposition_energy) / total_energy

# =============================================================================
# FIGURE
# =============================================================================
fig, ax1 = plt.subplots(figsize=(10, 5))

# -------------------------------------------------------------------------
# ABSOLUTE ENERGY FLUXES
# -------------------------------------------------------------------------
ax1.plot(
    plot_dates,
    melt_energy,
    linewidth=1.5,
    label='Melt energy'
)

ax1.plot(
    plot_dates,
    subl_energy,
    linewidth=1.5,
    label='Sublimation'
)

ax1.plot(
    plot_dates,
    deposition_energy,
    linewidth=1.5,
    label='Deposition'
)

ax1.set_ylabel('Energy flux (W m$^{-2}$)')
ax1.set_xlabel('Fecha')

# =============================================================================
# SECONDARY AXIS: RELATIVE CONTRIBUTION
# =============================================================================
# ax2 = ax1.twinx()

# ax2.plot(
#     plot_dates,
#     melt_pct,
#     linestyle='--',
#     linewidth=1.0,
#     alpha=0.8,
#     label='Melt (%)'
# )

# ax2.plot(
#     plot_dates,
#     subl_pct,
#     linestyle='--',
#     linewidth=1.0,
#     alpha=0.8,
#     label='Sublimation (%)'
# )

# ax2.plot(
#     plot_dates,
#     deposition_pct,
#     linestyle='--',
#     linewidth=1.0,
#     alpha=0.8,
#     label='Deposition (%)'
# )

# ax2.set_ylabel('Relative contribution (%)')
# ax2.set_ylim(0, 100)

# =============================================================================
# FORMATTING
# =============================================================================
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

fig.autofmt_xdate()

plt.title('Energy Partitioning - Horcones Superior')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc='upper left',
    frameon=False
)

plt.grid(False)

plt.tight_layout()
plt.show()


#%% ANALISIS DE 

# =========================================================
# IMPORTO DATASET HOBO
# =========================================================
hobo_all = pd.read_csv(
    r"C:\Users\ThinkPad\OneDrive\IANIGLA\Campañas\Resultados\HOBO\HOBO_HS_ALL.csv"
)