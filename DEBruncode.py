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
DATE_END   = '2026-02-28'

# DEBRIS TRANSITION SITES
# DATE_START = '2024-02-18'
# DATE_END   = '2026-02-28'

# DEBRIS FREE SITES
# DATE_START = '2024-02-18'
# DATE_END   = '2026-02-28'

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
OUTPUT_DIR = r"C:\Users\ThinkPad\OneDrive\IANIGLA\PublicacionBalanceEnergia\output_debmodel\outputs_2023_2026_all"

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
L_s    = CONSTANTS["L_s"]
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

    # --- Diagnóstico completo de flujos (todas las ramas) --------------------
    Snet_array    = np.full(nt, np.nan)   # SW neta
    Ldown_array   = np.full(nt, np.nan)   # LW entrante (pass-through forzante)
    Lup_array     = np.full(nt, np.nan)   # LW saliente
    H_array       = np.full(nt, np.nan)   # calor sensible (QH)
    LE_array      = np.full(nt, np.nan)   # calor latente  (QL)
    P_array       = np.full(nt, np.nan)   # flujo de calor por lluvia
    QC_array      = np.full(nt, np.nan)   # conductivo en superficie (solo detrito)
    totflux_array = np.full(nt, np.nan)   # hielo/nieve: flujo total a T_s convergida
                                          # detrito: residuo de cierre (~0, control)
    regime_array  = np.empty(nt, dtype=object)   # 'snow' / 'ice' / 'debris'

    sublimation_array = np.zeros(nt)
    deposition_array  = np.zeros(nt)
    sublimation_mwe_array      = np.zeros(nt)   # masa sublimada de HIELO (m w.e./h)
    snow_sublimation_mwe_array = np.zeros(nt)   # masa sublimada de NIEVE (m w.e./h)

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
        # CleanIceModel return order (9 values, verificado en CleanIceModel.py):
        #   (melt, T_s, Snet, Ldown, Lup, H, LE, P, totflux)
        #   index:   0    1    2     3    4   5   6  7     8
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

            Snet_array[t]    = float(snow_results[2])
            Ldown_array[t]   = float(snow_results[3])
            Lup_array[t]     = float(snow_results[4])
            H_array[t]       = float(snow_results[5])
            LE_out           = float(snow_results[6])
            LE_array[t]      = LE_out
            P_array[t]       = float(snow_results[7])
            totflux_array[t] = float(snow_results[8])
            regime_array[t]  = 'snow'
            
            T_s_snow = float(snow_results[1])

            # LE < 0: vapour leaves surface → sublimation/evaporation (mass loss)
            # LE > 0: vapour onto surface  → deposition/condensation (mass gain)

            if LE_out < 0:
                sublimation_array[t] = abs(LE_out)
                L_eff = L_s if T_s_snow < 0.0 else L_v
                snow_sublimation_mwe_array[t] = (-LE_out) * timestep / L_eff / 1000.0
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

            Snet_array[t]    = float(ice_results[2])
            Ldown_array[t]   = float(ice_results[3])
            Lup_array[t]     = float(ice_results[4])
            H_array[t]       = float(ice_results[5])
            LE_out           = float(ice_results[6])
            LE_array[t]      = LE_out
            P_array[t]       = float(ice_results[7])
            totflux_array[t] = float(ice_results[8])
            regime_array[t]  = 'ice'
            
            T_s_real = float(ice_results[1])   # T_s convergida (puede ser <0°C)

            # LE < 0: vapour leaves surface → sublimation/evaporation (mass loss)
            # LE > 0: vapour onto surface  → deposition/condensation (mass gain)
            if LE_out < 0:
                sublimation_array[t] = abs(LE_out)
                L_eff = L_s if T_s_real < 0.0 else L_v
                # kg/m2 == mm w.e.; /1000 -> m w.e.
                sublimation_mwe_array[t] = (-LE_out) * timestep / L_eff / 1000.0
            elif LE_out > 0:
                deposition_array[t]  = LE_out

            # T_d_in reset each step: bare ice has no debris thermal mass
            # to carry forward. (LOGIC #7: intentional)
            T_s_in           = T_f
            T_d_in           = np.linspace(T_s_in, T_f, N, dtype=np.float64)[:-1]
            # DIAGNÓSTICO: guardamos la T_s REAL convergida (puede ser < 0 °C
            # de noche desde FIX #A), no T_f fijo. Clave para auditar QH.
            T_s_array[t]     = float(ice_results[1])
            T_d_matrix[t, :] = np.full(N - 1, T_f)

        # ------------------------------------------------------------------
        # BRANCH 3: No snow, debris-covered site → DEBmodel
        # DEBmodel return order (9 values, verificado en DEBmodel.py):
        #   (melt, T_s_out, T_d_out, Snet, Lup, G, H, LE, P)
        #   index:   0      1        2      3     4    5  6  7   8
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

            Snet_array[t]  = float(results[3])
            Ldown_array[t] = Ldown                    # pass-through del forzante
            Lup_array[t]   = float(results[4])
            QC_array[t]    = float(results[5])
            H_array[t]     = float(results[6])
            LE_out         = float(results[7])
            LE_array[t]    = LE_out
            P_array[t]     = float(results[8])
            # Residuo de cierre del balance en superficie (debería ser ~0;
            # si no lo es, el Newton-Raphson no convergió bien ese paso)
            totflux_array[t] = (Snet_array[t] + Ldown + Lup_array[t] +
                                QC_array[t] + H_array[t] + LE_out + P_array[t])
            regime_array[t]  = 'debris'

            debmelt_array[t]   = melt_t
            totalmelt_array[t] = melt_t

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
    # Energía de fusión equivalente al melt del timestep (W m-2) — comparable
    # entre ramas: en hielo ≈ totflux de las horas de fusión; en detrito ≈ G_i
    # en la base del detrito (recorta los G_i < 0 igual que el melt).
    Qmelt_array = totalmelt_array * rho_w * L_f / timestep

    tbl_base = pd.DataFrame({
        'Fecha'            : pd.to_datetime(dates),
        'regime'           : regime_array,
        'TotalMelt_m_we'   : totalmelt_array,
        'SnowMelt_m_we'    : snowmelt_array,
        'DebrisMelt_m_we'  : debmelt_array,
        'IceMelt_m_we'     : icemelt_array,
        'T_s_surface_C'    : T_s_array,
        'Snet_W_m2'        : Snet_array,
        'Ldown_W_m2'       : Ldown_array,
        'Lup_W_m2'         : Lup_array,
        'H_W_m2'           : H_array,
        'LE_W_m2'          : LE_array,
        'P_W_m2'           : P_array,
        'QC_W_m2'          : QC_array,
        'TotFlux_W_m2'     : totflux_array,
        'Qmelt_W_m2'       : Qmelt_array,
        'Sublimation_W_m2' : sublimation_array,
        'Sublimation_m_we'     : sublimation_mwe_array,       # solo hielo
        'SnowSublimation_m_we' : snow_sublimation_mwe_array,  # solo nieve
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
        hourly_series = icemelt_array + sublimation_mwe_array
        hourly_label  = 'Ice ablation: melt + subl (m w.e./h)'
        hourly_color  = 'steelblue'
    else:
        hourly_series = debmelt_array
        hourly_label  = 'Debris melt (m w.e./h)'
        hourly_color  = 'brown'
   
    summary[site_name] = {
    "dates"    : plot_dates,
    "cum_main" : np.cumsum(hourly_series),
    "cum_snow" : np.cumsum(snowmelt_array),
    "cum_melt_only" : np.cumsum(icemelt_array if is_ice_site else debmelt_array),
    "cum_subl"      : np.cumsum(sublimation_mwe_array) if is_ice_site else None,   

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


    # ---------------------------------------------------------------------
    # GAP ENVELOPE — reconstruccion min/max del melt en huecos de forzante
    #   min: gaps aportan 0 melt (comportamiento actual del modelo)
    #   max: cada dia gap sin nieve se rellena con el melt medio de los
    #        dias validos en una ventana de +/-15 dias.
    #   Dias gap con nieve segun ANDESMAG (dato satelital, disponible aun
    #   sin meteo) NO se rellenan: melt de detrito ~0 bajo nieve.
    # ---------------------------------------------------------------------
    def gap_envelope(dates, cum_main, site_name,
                     window_days=15, min_hours_valid=20):

        melt_h = pd.Series(
            np.diff(np.asarray(cum_main), prepend=0.0),
            index=pd.DatetimeIndex(dates)
        )

        cal = pd.date_range(pd.Timestamp(DATE_START).normalize(),
                            pd.Timestamp(DATE_END).normalize(), freq='D')

        daily_melt  = melt_h.resample('D').sum().reindex(cal, fill_value=0.0)
        hours_valid = melt_h.resample('D').count().reindex(cal, fill_value=0)
        gap_day     = hours_valid < min_hours_valid

        # Mascara de nieve diaria desde el met_data (ANDESMAG cubre el gap)
        met_env = pd.read_csv(
            os.path.join(INPUT_DIR, f"met_data_{site_name}.csv"),
            parse_dates=['date']).set_index('date')
        snow_daily = (met_env['snow_presence'].fillna(0)
                      .resample('D').mean()
                      .reindex(cal).fillna(0))
        snowy_day = snow_daily >= 0.5   # dia nevado = mayoria de horas con flag

        # Tasa "tipica" local: media movil centrada de los dias validos
        # (P75 en vez de media: cambiar .mean() por .quantile(0.75))
        valid_melt = daily_melt.where(~gap_day)
        typical = (valid_melt
                   .rolling(f'{2*window_days + 1}D', center=True, min_periods=5)
                   .mean()
                   .fillna(0.0)
                   .clip(lower=0.0))

        # Relleno solo en dias gap SIN nieve; nunca por debajo del parcial ya medido
        fill = np.where(gap_day & ~snowy_day,
                        np.maximum(typical.values - daily_melt.values, 0.0),
                        0.0)

        cum_min = daily_melt.cumsum().values
        cum_max = (daily_melt + fill).cumsum().values

        n_filled = int((gap_day & ~snowy_day).sum())
        n_snowgap = int((gap_day & snowy_day).sum())
        print(f"    [{site_name}] gap envelope: {n_filled} dias rellenados, "
              f"{n_snowgap} dias gap con nieve (no rellenados) | "
              f"melt extra max = {cum_max[-1] - cum_min[-1]:.3f} m w.e.")

        return cal, cum_min, cum_max

    fig, ax = plt.subplots(figsize=(6.85, 3.5))

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
            linestyle='-',
            alpha=0.5
        )
        
        # --- Abanico min/max por gaps de forzante ---
        cal_env, cum_min_env, cum_max_env = gap_envelope(
            s_data["dates"], s_data["cum_main"], site_name
        )
        ax.fill_between(
            cal_env, cum_min_env, cum_max_env,
            color=color, alpha=0.25, linewidth=0, zorder=2,
            label=f'{site_name} gap-fill max'
        )
        # Borde superior del abanico con el mismo estilo que la serie
        ax.plot(
            cal_env, cum_max_env,
            color=color, linewidth=1.5, linestyle='-', alpha=0.5,
            zorder=3
        )

        site_stakes = stakes_df[
            stakes_df['site_norm'] == site_name
        ].copy()

        if not site_stakes.empty:

            # Balizas perdidas (h_m == 3): el valor medido es un MINIMO
            # confiable; el maximo es incierto -> barra de error solo
            # hacia arriba (asimetrica), no +-.
            err  = site_stakes['cum_ablation_err_mweq'].to_numpy(dtype=float)
            lost = np.isclose(site_stakes['h_m'].to_numpy(dtype=float), 3.0)

            yerr = np.vstack([
                np.where(lost, 0.0, err),   # rama inferior: 0 si es perdida
                err,                        # rama superior: siempre
            ])

            ax.errorbar(
                site_stakes['end_date'],
                site_stakes['cum_ablation_mweq'],
                yerr=yerr,
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
    ax.set_ylabel('Cumulative ablation (m w.e.)')
    ax.set_ylim(bottom=0)

    ax.set_title(
        f'Ablación acumulada  |  '
        f'Horcones Superior {period_str.replace("_", "–")}'
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    # ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.legend("")

    ax.grid(False)

    # ---------------------------------------------------------------------
    # ADD PARAMETERIZATION TEXT
    # ---------------------------------------------------------------------
    textstr = '\n'.join(text_lines)

    ax.text(
        0.015,
        0.95,
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