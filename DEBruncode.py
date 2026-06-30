# -*- coding: utf-8 -*-
"""
DEBruncode.py
--------------------------
Runs the debris energy balance (DEB) model and clean ice model for all
sites defined in config.py.

Site-type logic per timestep
-----------------------------
  snow_presence == 1  →  CleanIceModel  (SNOW parameters, all sites)
  d == 0.0            →  CleanIceModel  (site ICE parameters)
  d  > 0.0            →  DEBmodel       (site DEBRIS parameters)

Fixes applied vs v2
--------------------
  BUG #1  : rho_i (not rho_w) now passed to CleanIceModel in both
             snow and ice branches.
  BUG #2  : t_a_k column is assumed to be in Kelvin; converted to °C
             (T_a_C) before the Magnus-Tetens formula and before being
             passed to both model functions.
  BUG #3  : dropna restricted to truly critical met columns only;
             non-critical columns (rh_sfc_per, d_m, r_mm, snow_presence)
             filled with safe defaults instead of being dropped.
  LOGIC #4: T_d columns omitted from output CSV for clean-ice sites
             (they carry no physical meaning there).
  LOGIC #5: Date filter, filenames, and plot titles now all derive from
             the same DATE_START / DATE_END variables — no more mismatch.
  LOGIC #6: WARNING message uses the same dynamic period string.
  LOGIC #7: Ice-site T_d_in reset behaviour documented with a comment.
  LOGIC #8: CleanIceModel return-order verified in the docstring below;
             index [5] maps to LE_out (melt, Snet, Ldown_out, Lup,
             H, LE, P).

Author: adapted from original single-site script (2025)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

from DEBmodel import DEBmodel
from CleanIceModel import CleanIceModel
from config import CONSTANTS, MODEL, SNOW, SITES

# =============================================================================
# DATE RANGE
# Edit DATE_START and DATE_END to control the run period.
# Any format that pandas.Timestamp understands is accepted, e.g.:
#   '2023-01-01', '2023-06-15 00:00:00', '2024'
# =============================================================================
# DEBRIS COVER SITES
DATE_START = '2023-02-22'
DATE_END   = '2026-02-18'

# DEBRIS TRANSITION SITES
# DATE_START = '2024-02-18'
# DATE_END   = '2026-02-18'

# DEBRIS FREE SITES
# DATE_START = '2024-02-18'
# DATE_END   = '2026-02-19'

# Derive a compact string used in filenames and plot titles
_ts  = pd.Timestamp(DATE_START)
_te  = pd.Timestamp(DATE_END)
yr_start   = _ts.year
yr_end     = _te.year
period_str = f"{yr_start}_{yr_end}" if yr_start != yr_end else str(yr_start)

# =============================================================================
# PATHS
# =============================================================================
INPUT_DIR  = r"C:\Users\ThinkPad\OneDrive\IANIGLA\PublicacionBalanceEnergia\input_debmodel\primary_interp"
OUTPUT_DIR = r"C:\Users\ThinkPad\OneDrive\IANIGLA\PublicacionBalanceEnergia\output_debmodel"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# UNPACK UNIVERSAL CONSTANTS  (from config.py)
# =============================================================================
g      = CONSTANTS["g"]
k_vk   = CONSTANTS["k_vk"]
sigma  = CONSTANTS["sigma"]
Rgas   = CONSTANTS["Rgas"]
Mair   = CONSTANTS["Mair"]
p0     = CONSTANTS["p0"]
T0     = CONSTANTS["T0"]
Lapse  = CONSTANTS["Lapse"]
L_v    = CONSTANTS["L_v"]
L_f    = CONSTANTS["L_f"]
rho_w  = CONSTANTS["rho_w"]
c_w    = CONSTANTS["c_w"]
c_ad   = CONSTANTS["c_ad"]
rho_i  = CONSTANTS["rho_i"]
T_f    = CONSTANTS["T_f"]

# =============================================================================
# UNPACK MODEL SETTINGS  (from config.py)
# =============================================================================
timestep = MODEL["timestep"]
z_a      = MODEL["z_a"]
range_   = MODEL["range_"]
N        = MODEL["N"]
if N < 10:
    N = 10

# =============================================================================
# UNPACK SNOW PARAMETERS  (from config.py)
# =============================================================================
z_0_s     = SNOW["z_0_s"]
albedo_s  = SNOW["albedo_s"]
epsilon_s = SNOW["epsilon_s"]

# =============================================================================
# CRITICAL MET COLUMNS (used for dropna — non-critical cols handled below)
# =============================================================================
CRITICAL_VARS = [
    't_a_k',      # air temperature (K)
    'rh_a_per',   # relative humidity of air (%)
    'u_ms',       # wind speed (m/s)
    'sdown_wm2',  # downwelling shortwave (W/m²)
    'ldown_wm2',  # downwelling longwave  (W/m²)
    'p_a_pa',     # atmospheric pressure  (Pa)
]

# =============================================================================
# STORAGE FOR SUMMARY PLOT (cumulative melt per site)
# =============================================================================
summary = {}   # site_name → {"dates": ..., "cum_total": ...}

# =============================================================================
# MAIN LOOP OVER SITES
# =============================================================================
for site_name, site_cfg in SITES.items():

    print(f"\n{'='*60}")
    print(f"  Processing site: {site_name}  ({period_str})")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # 1. LOAD & PREPARE DATA
    # -------------------------------------------------------------------------
    meteo_path = os.path.join(INPUT_DIR, site_cfg["meteo_file"])
    data = pd.read_csv(meteo_path, na_values=['NaN', 'NA', 'null', '', ' ', 'nan', 'N/A'])

    # Parse datetime — let pandas infer format to be robust
    data['date'] = pd.to_datetime(data['date'])

    # Apply date range filter (inclusive on both ends)
    mask = (data['date'] >= DATE_START) & (data['date'] <= DATE_END)
    data = data[mask].copy()

    # --- BUG #3 FIX: drop only on critical met variables ---
    missing_critical = [c for c in CRITICAL_VARS if c not in data.columns]
    if missing_critical:
        print(f"  ERROR: Critical columns missing from CSV: {missing_critical}")
        print(f"  Skipping site {site_name}.")
        continue

    data = data.dropna(subset=CRITICAL_VARS).reset_index(drop=True)

    # Fill non-critical columns with safe defaults
    if 'r_mm' in data.columns:
        data['r_mm'] = data['r_mm'].fillna(0.0)
    else:
        data['r_mm'] = 0.0

    if 'snow_presence' in data.columns:
        data['snow_presence'] = data['snow_presence'].fillna(0).astype(int)
    else:
        data['snow_presence'] = 0

    if 'rh_sfc_per' in data.columns:
        # Cap at 100 % and fill gaps with saturation (conservative assumption)
        data['rh_sfc_per'] = data['rh_sfc_per'].clip(upper=100.0).fillna(100.0)
    else:
        data['rh_sfc_per'] = None

    dates = data['date'].values   # numpy datetime64 array
    
    dt = pd.Series(dates).diff()

    # print(dt.value_counts().head(10))
    
    nt    = len(data)

    # --- LOGIC #6 FIX: warning message uses dynamic period_str ---
    if nt == 0:
        print(f"  WARNING: No valid data for {site_name} in {period_str}. Skipping.")
        continue

    print(f"  Loaded {nt} valid timesteps.")

    # -------------------------------------------------------------------------
    # 2. IDENTIFY SITE TYPE
    # -------------------------------------------------------------------------
    d_config    = float(site_cfg["d"])
    is_ice_site = (d_config == 0.0)

    # -------------------------------------------------------------------------
    # 3. UNPACK SITE-SPECIFIC PARAMETERS
    # -------------------------------------------------------------------------
    z_0_d = float(site_cfg["z_0_d"])

    if is_ice_site:
        epsilon_i = float(site_cfg["epsilon_i"])
        albedo_i  = float(site_cfg["albedo_i"])
        # Debris params not applicable — set to NaN to catch accidental use
        k_d = c_d = rho_d = epsilon_d = albedo_d = np.nan
    else:
        k_d       = float(site_cfg["k_d"])
        rho_d     = float(site_cfg["rho_d"])
        c_d       = float(site_cfg["c_d"])
        epsilon_d = float(site_cfg["epsilon_d"])
        albedo_d  = float(site_cfg["albedo_d"])
        epsilon_i = albedo_i = np.nan

    # -------------------------------------------------------------------------
    # 4. INITIALISE OUTPUT ARRAYS
    # -------------------------------------------------------------------------
    snowmelt_array    = np.zeros(nt)
    debmelt_array     = np.zeros(nt)
    icemelt_array     = np.zeros(nt)
    totalmelt_array   = np.zeros(nt)
    T_s_array         = np.full(nt, np.nan)
    T_d_matrix        = np.full((nt, N - 1), np.nan)
    LE_array          = np.full(nt, np.nan)
    sublimation_array = np.zeros(nt)
    deposition_array  = np.zeros(nt)

    # -------------------------------------------------------------------------
    # 5. INITIAL CONDITIONS
    # -------------------------------------------------------------------------
    # BUG #2: t_a_k is in Kelvin → convert to °C for model initialisation
    T_a_first_C = float(data['t_a_k'].iloc[0]) - 273.15

    if is_ice_site:
        # Ice surface is assumed to start at melting point
        T_s_in = T_f
    else:
        # Debris surface starts at first air temperature (°C)
        T_s_in = T_a_first_C

    # Linear temperature profile from surface to ice–debris interface (T_f)
    T_d_in = np.linspace(T_s_in, T_f, N, dtype=np.float64)[:-1]  # shape (N-1,)

    # -------------------------------------------------------------------------
    # 6. TIME LOOP
    # -------------------------------------------------------------------------
    for t in range(nt):

        # --- BUG #2 FIX: convert Kelvin → °C before any calculation ---
        T_a_K  = float(data['t_a_k'].iloc[t])
        T_a_C  = T_a_K - 273.15   # used in Magnus-Tetens and passed to models

        # ------------------------------------------------------------------
        # RESET THERMAL PROFILE AFTER LARGE TEMPORAL GAPS
        # ------------------------------------------------------------------
        # The DEB model assumes approximately continuous forcing.
        # Large temporal gaps (days) can leave the propagated debris
        # temperature profile physically inconsistent with the new forcing.
        #
        # After a large gap, reinitialize:
        #   - surface temperature
        #   - internal debris temperature profile
        #
        # Small gaps (<= 3 h) were already interpolated previously.
        # ------------------------------------------------------------------

        if t > 0:

            dt_hours = (
                pd.Timestamp(dates[t]) -
                pd.Timestamp(dates[t - 1])
            ).total_seconds() / 3600.0

            if dt_hours > 3:

                # print(
                #     f"\nLarge temporal gap detected "
                #     f"at timestep {t}: "
                #     f"{dt_hours:.1f} hours"
                # )

                # print(
                #     f"  Previous date: {dates[t - 1]}"
                # )

                # print(
                #     f"  Current  date: {dates[t]}"
                # )

                # Reinitialize surface temperature
                T_s_in = T_a_C

                # Reinitialize debris thermal profile
                T_d_in = np.linspace(
                    T_s_in,
                    T_f,
                    N,
                    dtype=np.float64
                )[:-1]

        Sdown  = float(data['sdown_wm2'].iloc[t])
        Ldown  = float(data['ldown_wm2'].iloc[t])
        u      = float(data['u_ms'].iloc[t])
        RH_a   = float(data['rh_a_per'].iloc[t])
        RH_sfc = float(data['rh_sfc_per'].iloc[t])   # already capped at 100
        r      = float(data['r_mm'].iloc[t]) / (1000.0 * timestep)   # mm/hr → m/s
        p_a    = float(data['p_a_pa'].iloc[t])

        # Derived humidity — Magnus-Tetens in °C (BUG #2 fix)
        e_s = float(610.8 * np.exp(17.27 * T_a_C / (237.3 + T_a_C)))
        e_a = float(RH_a * e_s / 100.0)
        q_a = float(0.622 * e_a / (p_a - 0.378 * e_a))

        # Ensure T_d_in is a clean float64 array
        T_d_in = np.array(T_d_in, dtype=np.float64)
        T_s_in = float(T_s_in)

        # ------------------------------------------------------------------
        # BRANCH 1: Snow present → CleanIceModel with SNOW parameters
        # Applies to ALL site types when snow_presence == 1.
        # CleanIceModel return order (9 values):
        #   (melt, T_s, Snet, Ldown, Lup, H, LE, P, totflux)
        #   index:   0    1    2     3     4   5   6   7     8
        #   → latent heat flux (LE) is index [6]; [5] is sensible heat (H)
        # ------------------------------------------------------------------
        if int(data['snow_presence'].iloc[t]) == 1:

            snow_results = CleanIceModel(
                timestep, Sdown, Ldown, T_a_C, u, q_a, r,
                albedo_s, epsilon_s, z_0_s,
                g, k_vk, sigma, Rgas, Mair, L_v, L_f,
                rho_w, c_w, c_ad, rho_i, T_f, p_a, z_a   # BUG #1 FIX: rho_i
            )

            snowmelt_array[t]  = float(snow_results[0])
            totalmelt_array[t] = snowmelt_array[t]

            LE_out          = float(snow_results[6])       # latent heat flux (LE), index [6]
            LE_array[t]     = LE_out
            # LE < 0: vapour leaves surface → sublimation/evaporation (mass loss)
            # LE > 0: vapour onto surface  → deposition/condensation (mass gain)
            if LE_out < 0:
                sublimation_array[t] = abs(LE_out)
            elif LE_out > 0:
                deposition_array[t]  = LE_out

            # Snow covers the debris — reset surface and profile
            T_s_in = T_a_C
            T_d_in = np.linspace(T_s_in, T_f, N, dtype=np.float64)[:-1]

            # Surface and internal temperatures undefined under snow
            T_s_array[t]     = np.nan
            T_d_matrix[t, :] = np.nan

        # ------------------------------------------------------------------
        # BRANCH 2: No snow, clean-ice site → CleanIceModel with ICE params
        # ------------------------------------------------------------------
        elif is_ice_site:

            ice_results = CleanIceModel(
                timestep, Sdown, Ldown, T_a_C, u, q_a, r,
                albedo_i, epsilon_i, z_0_d,
                g, k_vk, sigma, Rgas, Mair, L_v, L_f,
                rho_w, c_w, c_ad, rho_i, T_f, p_a, z_a   # BUG #1 FIX: rho_i
            )

            icemelt_array[t]   = float(ice_results[0])
            totalmelt_array[t] = icemelt_array[t]

            LE_out          = float(ice_results[6])        # latent heat flux (LE), index [6]
            LE_array[t]     = LE_out
            # LE < 0: vapour leaves surface → sublimation/evaporation (mass loss)
            # LE > 0: vapour onto surface  → deposition/condensation (mass gain)
            if LE_out < 0:
                sublimation_array[t] = abs(LE_out)
            elif LE_out > 0:
                deposition_array[t]  = LE_out

            # Ice surface is fixed at melting point.
            # T_d_in is reset to all-T_f each step because bare ice has no
            # debris thermal mass to carry forward. (LOGIC #7: intentional)
            T_s_in           = T_f
            T_d_in           = np.linspace(T_s_in, T_f, N, dtype=np.float64)[:-1]
            T_s_array[t]     = T_f
            T_d_matrix[t, :] = np.full(N - 1, T_f)

        # ------------------------------------------------------------------
        # BRANCH 3: No snow, debris-covered site → DEBmodel
        # ------------------------------------------------------------------
        else:

            results = DEBmodel(
                timestep, Sdown, Ldown, T_a_C, u, q_a, RH_sfc, r,
                T_s_in, T_d_in, d_config, z_a, z_0_d,
                k_d, rho_d, c_d, epsilon_d, albedo_d, range_, N,
                g, k_vk, sigma, Rgas, Mair, L_v, L_f,
                rho_w, c_w, c_ad, rho_i, T_f, p_a
            )

            melt_t  = float(results[0])
            T_s_out = float(results[1])
            T_d_out = np.array(results[2], dtype=np.float64)
            LE_out  = float(results[7])

            debmelt_array[t]   = melt_t
            totalmelt_array[t] = melt_t

            LE_array[t] = LE_out
            # LE < 0: vapour leaves surface → sublimation/evaporation (mass loss)
            # LE > 0: vapour onto surface  → deposition/condensation (mass gain)
            if LE_out < 0:
                sublimation_array[t] = abs(LE_out)
            elif LE_out > 0:
                deposition_array[t]  = LE_out

            T_s_array[t]     = T_s_out
            T_d_matrix[t, :] = T_d_out

            # Carry forward for next timestep
            T_s_in = T_s_out
            T_d_in = T_d_out
            
    
    # -------------------------------------------------------------------------
    # 7. EXPORT RESULTS
    # -------------------------------------------------------------------------
    tbl_base = pd.DataFrame({
        'Fecha'            : pd.to_datetime(dates),
        'TotalMelt_m_we'   : totalmelt_array,
        'SnowMelt_m_we'    : snowmelt_array,
        'DebrisMelt_m_we'  : debmelt_array,
        'IceMelt_m_we'     : icemelt_array,
        'T_s_surface_C'    : T_s_array,
        'LE_W_m2'          : LE_array,
        'Sublimation_W_m2' : sublimation_array,
        'Deposition_W_m2'  : deposition_array,
    })

    # LOGIC #4 FIX: T_d columns only written for debris sites
    if not is_ice_site:
        td_names = [f'T_d_{i+1}' for i in range(N - 1)]
        tbl_td   = pd.DataFrame(T_d_matrix, columns=td_names)
        results_table = pd.concat([tbl_base, tbl_td], axis=1)
    else:
        results_table = tbl_base

    # LOGIC #5 FIX: filename derived from DATE_START / DATE_END
    out_csv = os.path.join(OUTPUT_DIR, f'outputs_{site_name}_{period_str}.csv')
    results_table.to_csv(out_csv, index=False)
    print(f"  Results saved → {out_csv}")

    # -------------------------------------------------------------------------
    # 8. PER-SITE PLOT: hourly melt + cumulative total melt
    # -------------------------------------------------------------------------

    # --- Journal of Glaciology double-column figure settings ---
    plt.rcParams.update({
        'font.size'       : 9,
        'axes.titlesize'  : 10,
        'axes.labelsize'  : 9,
        'xtick.labelsize' : 8,
        'ytick.labelsize' : 8,
        'legend.fontsize' : 8,
    })

    plot_dates = pd.to_datetime(dates)
    cum_total  = np.cumsum(totalmelt_array)

    if is_ice_site:
        hourly_series = icemelt_array
        hourly_label  = 'Ice melt (m w.e./h)'
        hourly_color  = 'steelblue'
    else:
        hourly_series = debmelt_array
        hourly_label  = 'Debris melt (m w.e./h)'
        hourly_color  = 'brown'

    fig, ax1 = plt.subplots(figsize=(6.85, 4.5))

    ax1.set_xlabel('Fecha')
    ax1.set_ylabel(hourly_label, color=hourly_color)

    ax1.plot(plot_dates, hourly_series,
             color=hourly_color,
             linewidth=0.8,
             alpha=0.9,
             label=hourly_label)

    ax1.plot(plot_dates, snowmelt_array,
             color='lightblue',
             linewidth=0.6,
             alpha=1,
             label='Snow melt (m w.e./h)')

    ax1.tick_params(axis='y', labelcolor=hourly_color)

    ax1.set_xlim(pd.to_datetime(DATE_START),
                 pd.to_datetime(DATE_END))

    ax2 = ax1.twinx()

    cum_snow = np.cumsum(snowmelt_array)

    if is_ice_site:

        cum_ice   = np.cumsum(icemelt_array)
        cum_stack = cum_ice + cum_snow

        ax2.set_ylabel('Cumulative melt (m w.e.)')

        ax2.fill_between(plot_dates, 0,       cum_ice,   color='lightblue',   alpha=0.2, label='Cum. ice melt')
        ax2.fill_between(plot_dates, cum_ice,  cum_stack, color='cyan',        alpha=0.2, label='Cum. snow melt')

    else:

        cum_debris = np.cumsum(debmelt_array)
        cum_stack  = cum_debris + cum_snow

        ax2.set_ylabel('Cumulative melt (m w.e.)')

        ax2.fill_between(plot_dates, 0,          cum_debris, color='tab:orange', alpha=0.2, label='Cum. debris melt')
        ax2.fill_between(plot_dates, cum_debris,  cum_stack,  color='cyan',       alpha=0.2, label='Cum. snow melt')

    ax2.tick_params(axis='y')
    ax2.set_ylim(bottom=0)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    plt.title(f'Ablación Horcones Superior — {site_name}  ({period_str.replace("_", "–")})')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    plt.grid(False)
    plt.tight_layout()

    fig_path = os.path.join(OUTPUT_DIR, f'plot_{site_name}_{period_str}.png')
    plt.savefig(fig_path, dpi=300)
    plt.show()
    plt.close()

    print(f"  Plot saved    → {fig_path}")

    # summary[site_name] = {
    #     "dates"    : plot_dates,
    #     "cum_main" : np.cumsum(hourly_series),
    #     "cum_snow" : np.cumsum(snowmelt_array),
    # }
    
    summary[site_name] = {
    "dates"    : plot_dates,
    "cum_main" : np.cumsum(hourly_series),
    "cum_snow" : np.cumsum(snowmelt_array),

    # --- site parameters from config ---
    "is_ice_site" : is_ice_site,
    "z_0_d"       : z_0_d,

    # debris sites
    "albedo_d"    : albedo_d if not is_ice_site else np.nan,
    "k_d"         : k_d if not is_ice_site else np.nan,
    "rho_d" : rho_d if not is_ice_site else np.nan, 

    # ice sites
    "albedo_i"    : albedo_i if is_ice_site else np.nan,
}

# =============================================================================
# SUMMARY PLOT: cumulative melt for all sites
# =============================================================================
if summary:
    print("\nGenerating summary plot …")

    # --- Journal of Glaciology double-column figure settings ---
    plt.rcParams.update({
        'font.size'       : 9,
        'axes.titlesize'  : 10,
        'axes.labelsize'  : 9,
        'xtick.labelsize' : 8,
        'ytick.labelsize' : 8,
        'legend.fontsize' : 8,
    })

    PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c',
               '#d62728', '#9467bd', '#8c564b']

    stakes_path = os.path.join(
        r"C:\Users\ThinkPad\OneDrive\IANIGLA\PublicacionBalanceEnergia",
        'hs_ablation_stakes_data.xlsx'
    )

    stakes_df = pd.read_excel(stakes_path)

    stakes_df['end_date']  = pd.to_datetime(stakes_df['end_date'])
    stakes_df['site_norm'] = stakes_df['site'].str.replace(
        r'_C$', '', regex=True
    )

    fig, ax = plt.subplots(figsize=(6.85, 4.5))

    # ---------------------------------------------------------------------
    # TEXT BLOCK WITH SITE PARAMETERIZATION
    # ---------------------------------------------------------------------
    text_lines = []

    for (site_name, s_data), color in zip(summary.items(), PALETTE):

        ax.plot(
            s_data["dates"],
            s_data["cum_main"],
            label=f'{site_name} model',
            color=color,
            linewidth=1.5,
            linestyle='-'
        )

        site_stakes = stakes_df[
            stakes_df['site_norm'] == site_name
        ].copy()

        if not site_stakes.empty:

            ax.errorbar(
                site_stakes['end_date'],
                site_stakes['cum_ablation_mweq'],
                yerr=site_stakes['cum_ablation_err_mweq'],
                fmt='o',
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=3,
                markersize=5,
                label=f'{site_name} stakes',
                zorder=5
            )

        # -----------------------------------------------------------------
        # PARAMETER TEXT
        # -----------------------------------------------------------------
        if s_data["is_ice_site"]:

            txt = (
                f'{site_name}: '
                f'albedo_i={s_data["albedo_i"]:.2f}, '
                f'z_0_d={s_data["z_0_d"]:.4f}'
            )

        else:

            txt = (
                f'{site_name}: '
                f'albedo_d={s_data["albedo_d"]:.2f}, '
                f'z_0_d={s_data["z_0_d"]:.4f}, '
                f'k_d={s_data["k_d"]:.2f} '
                f'rho_d ={s_data["rho_d"]:.2f}'
            )

        text_lines.append(txt)

    # ---------------------------------------------------------------------
    # AXES
    # ---------------------------------------------------------------------
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Cumulative melt (m w.e.)')
    ax.set_ylim(bottom=0)

    ax.set_title(
        f'Ablación acumulada  |  '
        f'Horcones Superior {period_str.replace("_", "–")}'
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    ax.legend(loc='upper left', fontsize=8, ncol=2)

    ax.grid(True, alpha=0.3)

    # ---------------------------------------------------------------------
    # ADD PARAMETERIZATION TEXT
    # ---------------------------------------------------------------------
    textstr = '\n'.join(text_lines)

    ax.text(
        0.015,
        0.85,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.3
        )
    )

    plt.tight_layout()

    # ---------------------------------------------------------------------
    # DYNAMIC FILENAME INCLUDING SITE NAMES
    # ---------------------------------------------------------------------
    site_suffix = "_".join(summary.keys())

    summary_fig_path = os.path.join(
        OUTPUT_DIR,
        f'plot_ALL_sites_cumulative_{period_str}_{site_suffix}.png'
    )

    plt.savefig(
        summary_fig_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    plt.close()

    print(f"Summary plot saved → {summary_fig_path}")

else:
    print("\nNo sites produced output — summary plot skipped.")

print("\nAll sites processed successfully.")