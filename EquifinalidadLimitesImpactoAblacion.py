# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 19:24:30 2026

@author: ThinkPad
"""

# %% ========================================================================
# LÍMITES DE ERROR POR EQUIFINALIDAD — extremos de k y z0 (m w.e.)  [v2: subl]
# ============================================================================
# Corre SOLO los extremos de la caja aceptable de cada sitio:
#   detrito -> óptimo + 4 esquinas (k_min/k_max × z0_min/z0_max)
#   hielo   -> óptimo + 2 extremos (z0_min, z0_max)   [albedo fijo de config]
# Para cada esquina: ablación acumulada (punto medio del gap-envelope) y su
# residuo vs balizas. La ablación es monótona en (k, z0), así que las esquinas
# ACOTAN el rango posible dentro de la equifinalidad.
#
# CORRECCIÓN (v2): en HIELO la ablación = fusión + sublimación/evaporación
# (las balizas miden descenso TOTAL de superficie). La versión anterior solo
# contaba fusión y subestimaba la ablación a z0 alto. Los rangos de z0 de hielo
# son los PLAUSIBLES del barrido v6 (aceptable y sublimación <= 30%), no el
# óptimo global z0=0.20 (que exigía 53% de sublimación).

import os, sys
import numpy as np
import pandas as pd

PYDEB_DIR   = r"C:\Users\ThinkPad\OneDrive\IANIGLA\PublicacionBalanceEnergia\scripts\PYDEB_github"
INPUT_DIR   = r"C:\Users\ThinkPad\OneDrive\IANIGLA\PublicacionBalanceEnergia\input_debmodel\primary_interp"
STAKES_XLSX = r"C:\Users\ThinkPad\OneDrive\IANIGLA\PublicacionBalanceEnergia\hs_ablation_stakes_data.xlsx"
OUT_DIR     = r"C:\Users\ThinkPad\OneDrive\IANIGLA\PublicacionBalanceEnergia\output_debmodel\extremos_equifinalidad"  # <<< CORRECCIÓN
os.makedirs(OUT_DIR, exist_ok=True)

if PYDEB_DIR not in sys.path:
    sys.path.insert(0, PYDEB_DIR)
from DEBmodel import DEBmodel
from CleanIceModel import CleanIceModel
from config import CONSTANTS, MODEL, SITES

g     = CONSTANTS["g"];     k_vk  = CONSTANTS["k_vk"];  sigma = CONSTANTS["sigma"]
Rgas  = CONSTANTS["Rgas"];  Mair  = CONSTANTS["Mair"]
L_v   = CONSTANTS["L_v"];   L_f   = CONSTANTS["L_f"];   L_s = CONSTANTS["L_s"]   # <<< CORRECCIÓN
rho_w = CONSTANTS["rho_w"]; c_w   = CONSTANTS["c_w"];   c_ad  = CONSTANTS["c_ad"]
rho_i = CONSTANTS["rho_i"]; T_f   = CONSTANTS["T_f"]
timestep = MODEL["timestep"]; z_a = MODEL["z_a"]
range_   = MODEL["range_"];   N   = max(MODEL["N"], 10)

CRITICAL_VARS = ['t_a_k', 'rh_a_per', 'u_ms', 'sdown_wm2', 'ldown_wm2', 'p_a_pa']
GAP_WINDOW_DAYS, GAP_MIN_HOURS_VALID, GAP_SNOW_FRAC = 15, 20, 0.5

# --- TABLA DE EQUIFINALIDAD (extremos): k y z0 por sitio --------------------
# claves con guion bajo (como config.py / balizas). Hielo: k=None.
# HIELO: rangos y óptimo PLAUSIBLES del barrido v6 (aceptable + subl <= 30%).
EQUIFIN = {
    "4400_DC": {"k": (0.954, 1.088), "z0": (0.0123, 0.0313), "opt": (0.991, 0.0123)},
    "4500_T" : {"k": (0.954, 1.051), "z0": (0.0128, 0.0203), "opt": (0.954, 0.0203)},
    "4500_DC": {"k": (1.265, 1.423), "z0": (0.0052, 0.0094), "opt": (1.423, 0.0052)},
    "4600_T" : {"k": (0.954, 1.088), "z0": (0.0471, 0.1596), "opt": (0.978, 0.0618)},
    "4500_I" : {"k": None, "z0": (0.0150, 0.0757), "opt": (None, 0.0319)},  # <<< CORRECCIÓN (v6 plausible)
    "4600_I" : {"k": None, "z0": (0.0150, 0.0680), "opt": (None, 0.0492)},  # <<< CORRECCIÓN (v6 plausible)
}

# =============================================================================
# FORZANTE + SIMULACIÓN + GAP ENVELOPE  (copiados del barrido v2/v3/v6)
# =============================================================================

def cargar_forzante(cfg, date_start, date_end):
    data = pd.read_csv(os.path.join(INPUT_DIR, cfg["meteo_file"]),
                       na_values=['NaN', 'NA', 'null', '', ' ', 'nan', 'N/A'])
    data['date'] = pd.to_datetime(data['date'])
    data = data[(data['date'] >= date_start) & (data['date'] <= date_end)].copy()

    cal = pd.date_range(pd.Timestamp(date_start).normalize(),
                        pd.Timestamp(date_end).normalize(), freq='D')
    if 'snow_presence' in data.columns:
        snow_daily = (data.set_index('date')['snow_presence'].fillna(0)
                      .resample('D').mean().reindex(cal).fillna(0))
    else:
        snow_daily = pd.Series(0.0, index=cal)
    snowy_day = (snow_daily >= GAP_SNOW_FRAC).values

    data = data.dropna(subset=CRITICAL_VARS).reset_index(drop=True)
    data['r_mm'] = data['r_mm'].fillna(0.0) if 'r_mm' in data else 0.0
    data['snow_presence'] = (data['snow_presence'].fillna(0).astype(int)
                             if 'snow_presence' in data else 0)
    data['rh_sfc_per'] = (data['rh_sfc_per'].clip(upper=100.0).fillna(100.0)
                          if 'rh_sfc_per' in data else 100.0)

    dates = data['date'].values
    T_a_C = data['t_a_k'].to_numpy(float) - 273.15
    p_a   = data['p_a_pa'].to_numpy(float)
    e_s   = 610.8 * np.exp(17.27 * T_a_C / (237.3 + T_a_C))
    e_a   = data['rh_a_per'].to_numpy(float) * e_s / 100.0
    q_a   = 0.622 * e_a / (p_a - 0.378 * e_a)
    dt_h  = np.diff(dates).astype('timedelta64[s]').astype(float) / 3600.0
    reset = np.concatenate([[False], dt_h > 3.0])

    hours_valid = (pd.Series(1.0, index=pd.DatetimeIndex(dates))
                   .resample('D').count().reindex(cal, fill_value=0))
    gap_day = (hours_valid < GAP_MIN_HOURS_VALID).values

    return {"dates": dates, "T_a_C": T_a_C,
            "Sdown": data['sdown_wm2'].to_numpy(float),
            "Ldown": data['ldown_wm2'].to_numpy(float),
            "u": data['u_ms'].to_numpy(float), "q_a": q_a,
            "RH_sfc": data['rh_sfc_per'].to_numpy(float),
            "r_ms": data['r_mm'].to_numpy(float) / (1000.0 * timestep),
            "p_a": p_a, "snow": data['snow_presence'].to_numpy(int),
            "reset": reset, "nt": len(data),
            "cal": cal, "cal_np": cal.values,
            "gap_day": gap_day, "snowy_day": snowy_day}


def simular_cum_debris(F, cfg, k_d, z_0_d):
    d = cfg["d"]; rho_d = cfg["rho_d"]; c_d = cfg["c_d"]
    eps_d = cfg["epsilon_d"]; albedo_d = cfg["albedo_d"]; nt = F["nt"]
    T_s_in = float(F["T_a_C"][0])
    T_d_in = np.linspace(T_s_in, T_f, N, dtype=np.float64)[:-1]
    cum, cum_arr = 0.0, np.empty(nt)
    for t in range(nt):
        Ta = float(F["T_a_C"][t])
        if F["reset"][t] or F["snow"][t] == 1:
            T_s_in = Ta
            T_d_in = np.linspace(Ta, T_f, N, dtype=np.float64)[:-1]
        if F["snow"][t] != 1:
            res = DEBmodel(
                timestep, float(F["Sdown"][t]), float(F["Ldown"][t]), Ta,
                float(F["u"][t]), float(F["q_a"][t]), float(F["RH_sfc"][t]),
                float(F["r_ms"][t]), T_s_in, T_d_in, d, z_a, z_0_d,
                k_d, rho_d, c_d, eps_d, albedo_d, range_, N,
                g, k_vk, sigma, Rgas, Mair, L_v, L_f,
                rho_w, c_w, c_ad, rho_i, T_f, float(F["p_a"][t]))
            cum += float(res[0]); T_s_in = float(res[1])
            T_d_in = np.array(res[2], dtype=np.float64)
        cum_arr[t] = cum
    return cum_arr


def simular_cum_ice(F, cfg, z_0_d):                       # <<< CORRECCIÓN (v6)
    """Ablación de hielo = fusión + sublimación/evaporación (como el v6).
    Devuelve (cum_total, fusión_total, sublimación_total)."""
    eps_i, albedo_i = cfg["epsilon_i"], cfg["albedo_i"]; nt = F["nt"]
    cum_m, cum_s, cum_arr = 0.0, 0.0, np.empty(nt)
    for t in range(nt):
        if F["snow"][t] != 1:
            res = CleanIceModel(
                timestep, float(F["Sdown"][t]), float(F["Ldown"][t]),
                float(F["T_a_C"][t]), float(F["u"][t]), float(F["q_a"][t]),
                float(F["r_ms"][t]), albedo_i, eps_i, z_0_d,
                g, k_vk, sigma, Rgas, Mair, L_v, L_f,
                rho_w, c_w, c_ad, rho_i, T_f, float(F["p_a"][t]), z_a)
            cum_m += float(res[0])
            LE_t, T_s_t = float(res[6]), float(res[1])        # LE y T_s
            if LE_t < 0.0:                                    # pérdida de masa
                L_eff = L_s if T_s_t < 0.0 else L_v
                cum_s += (-LE_t) * timestep / L_eff / 1000.0
        cum_arr[t] = cum_m + cum_s
    return cum_arr, cum_m, cum_s


def gap_envelope_run(F, cum_arr):
    melt_h = pd.Series(np.diff(cum_arr, prepend=0.0),
                       index=pd.DatetimeIndex(F["dates"]))
    daily = melt_h.resample('D').sum().reindex(F["cal"], fill_value=0.0)
    valid = daily.where(~F["gap_day"])
    typical = (valid.rolling(f'{2*GAP_WINDOW_DAYS + 1}D', center=True,
                             min_periods=5).mean().fillna(0.0).clip(lower=0.0))
    fill = np.where(F["gap_day"] & ~F["snowy_day"],
                    np.maximum(typical.values - daily.values, 0.0), 0.0)
    return daily.cumsum().values, (daily.values + fill).cumsum()

# =============================================================================
# BALIZAS
# =============================================================================

stakes_df = pd.read_excel(STAKES_XLSX)
stakes_df['start_date'] = pd.to_datetime(stakes_df['start_date'])
stakes_df['end_date']   = pd.to_datetime(stakes_df['end_date'])
stakes_df['site_norm']  = stakes_df['site'].str.replace(r'_C$', '', regex=True)
stakes_df['lost'] = np.isclose(stakes_df['h_m'].astype(float), 3.0)

def balizas_sitio(site):
    s = stakes_df[stakes_df['site_norm'] == site].sort_values('end_date')
    return None if s.empty else s[['start_date', 'end_date', 'cum_ablation_mweq',
                                   'cum_ablation_err_mweq', 'lost']].reset_index(drop=True)

def evaluar(F, cum_min, cum_max, balizas):
    """RMSE del punto medio del abanico vs balizas + detalle por lectura."""
    residuos, detalle = [], []
    for _, b in balizas.iterrows():
        idx = np.searchsorted(F["cal_np"],
                              np.datetime64(b['end_date'].normalize()), side='left')
        idx = min(idx, len(F["cal_np"]) - 1)
        s_mid = 0.5 * (cum_min[idx] + cum_max[idx])
        obs, err = b['cum_ablation_mweq'], b['cum_ablation_err_mweq']
        if b['lost']:
            lo, hi = obs, obs + err
            resid = (s_mid - lo if s_mid < lo else s_mid - hi if s_mid > hi else 0.0)
        else:
            resid = s_mid - obs
        residuos.append(resid)
        detalle.append({"end_date": b['end_date'], "obs": obs, "err": err,
                        "sim_mid": s_mid, "resid": resid})
    rmse = float(np.sqrt(np.mean(np.array(residuos) ** 2))) if residuos else np.nan
    return rmse, detalle

# =============================================================================
# CORRIDA DE EXTREMOS
# =============================================================================

def combos_sitio(eq, es_hielo):
    """Óptimo + esquinas de la caja de equifinalidad."""
    ko, zo = eq["opt"]
    if es_hielo:
        z0lo, z0hi = eq["z0"]
        return [("óptimo", None, zo),
                ("z0_min", None, z0lo), ("z0_max", None, z0hi)]
    klo, khi = eq["k"]; z0lo, z0hi = eq["z0"]
    return [("óptimo",       ko,  zo),
            ("k-,z0-", klo, z0lo), ("k-,z0+", klo, z0hi),
            ("k+,z0-", khi, z0lo), ("k+,z0+", khi, z0hi)]

filas, resumen = [], []
for site, eq in EQUIFIN.items():
    if site not in SITES:
        print(f"{site}: no está en config.SITES, se omite."); continue
    cfg = dict(SITES[site])                      # params fijos de config.py
    balizas = balizas_sitio(site)
    if balizas is None:
        print(f"{site}: sin balizas, se omite."); continue

    es_hielo = (cfg.get("d", 0.0) == 0.0)
    date_start = balizas['start_date'].min()
    date_end   = balizas['end_date'].max() + pd.Timedelta(days=1)
    F = cargar_forzante(cfg, date_start, date_end)

    print(f"\n{'='*80}\n  {site} — {'HIELO (melt+subl)' if es_hielo else 'DETRITO'} "
          f"| {len(balizas)} balizas | albedo fijo config="
          f"{cfg.get('albedo_i', cfg.get('albedo_d'))}\n{'='*80}")
    print(f"  {'combo':10s} {'k':>7s} {'z0':>8s}  {'abl_tot':>9s} "
          f"{'RMSE':>7s} {'resid_últ':>10s} {'subl%':>6s}   (m w.e.)")

    abl_list, err_list, rmse_list = [], [], []
    for label, k, z0 in combos_sitio(eq, es_hielo):
        if es_hielo:                                          # <<< CORRECCIÓN
            cum, tot_m, tot_s = simular_cum_ice(F, cfg, z0)
            fsubl = tot_s / (tot_m + tot_s) if (tot_m + tot_s) else np.nan
        else:
            cum = simular_cum_debris(F, cfg, k, z0)
            fsubl = np.nan
        cmin, cmax = gap_envelope_run(F, cum)
        rmse, det = evaluar(F, cmin, cmax, balizas)
        abl_tot   = 0.5 * (cmin[-1] + cmax[-1])      # ablación total (sim_mid)
        resid_ult = det[-1]['resid']                 # error última baliza

        k_txt = "   —  " if k is None else f"{k:6.3f}"
        s_txt = "   —  " if np.isnan(fsubl) else f"{100*fsubl:5.0f}%"
        print(f"  {label:10s} {k_txt} {z0:8.4f}  {abl_tot:9.3f} "
              f"{rmse:7.3f} {resid_ult:+10.3f} {s_txt:>6s}")
        filas.append({"site": site, "combo": label, "k": k, "z0": z0,
                      "abl_total_mwe": abl_tot, "rmse_mwe": rmse,
                      "resid_ultima_mwe": resid_ult, "frac_subl": fsubl})
        abl_list.append(abl_tot); err_list.append(resid_ult); rmse_list.append(rmse)

    abl = np.array(abl_list); err = np.array(err_list)
    print(f"  {'-'*72}")
    print(f"  ablación total   ∈ [{abl.min():.3f}, {abl.max():.3f}]  "
          f"Δ = {abl.max()-abl.min():.3f} m w.e.")
    print(f"  error (resid últ)∈ [{err.min():+.3f}, {err.max():+.3f}]  "
          f"ancho = {err.max()-err.min():.3f} m w.e.")
    resumen.append({"site": site,
                    "abl_min": abl.min(), "abl_opt": abl_list[0], "abl_max": abl.max(),
                    "abl_spread": abl.max()-abl.min(),
                    "err_min": err.min(), "err_max": err.max(),
                    "err_ancho": err.max()-err.min(),
                    "rmse_min": min(rmse_list), "rmse_max": max(rmse_list)})

df_extremos = pd.DataFrame(filas)
df_resumen  = pd.DataFrame(resumen)

print("\n" + "=" * 92)
print("RESUMEN — límites de error por equifinalidad (m w.e.)")
print("=" * 92)
print(df_resumen.round(3).to_string(index=False))

df_extremos.to_csv(os.path.join(OUT_DIR, "extremos_equifinalidad_completo.csv"), index=False)  # <<< CORRECCIÓN
df_resumen.to_csv(os.path.join(OUT_DIR, "extremos_equifinalidad_resumen.csv"), index=False)    # <<< CORRECCIÓN
print(f"\n✅ Guardado en {OUT_DIR}")



# %% ========================================================================
# ============================================================================
# ¿CUÁNTO NOS EQUIVOCAMOS SI NO OPTIMIZAMOS?
# ----------------------------------------------------------------------------
# Corre PyDEB por sitio con (a) parámetros óptimos del barrido y (b) las
# medianas medidas directamente (C&R para k, Rounce et al. 2015 para z0),
# y compara ablación acumulada y RMSE contra balizas (con gap envelope).
# Autocontenido: solo requiere PyDEB (DEBmodel/CleanIceModel/config) y los
# archivos de meteo y balizas.
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd

PYDEB_DIR   = r"C:\Users\ThinkPad\OneDrive\IANIGLA\PublicacionBalanceEnergia\scripts\PYDEB_github"
INPUT_DIR   = r"C:\Users\ThinkPad\OneDrive\IANIGLA\PublicacionBalanceEnergia\input_debmodel\primary_interp"
STAKES_XLSX = r"C:\Users\ThinkPad\OneDrive\IANIGLA\PublicacionBalanceEnergia\hs_ablation_stakes_data.xlsx"
OUT_DIR     = r"C:\Users\ThinkPad\OneDrive\IANIGLA\PublicacionBalanceEnergia\output_debmodel"

if PYDEB_DIR not in sys.path:
    sys.path.insert(0, PYDEB_DIR)

from DEBmodel import DEBmodel
from CleanIceModel import CleanIceModel
import importlib
import config
importlib.reload(config)
CONSTANTS, MODEL = config.CONSTANTS, config.MODEL

# =============================================================================
# CASOS A COMPARAR: (k, z0) en detrito | (albedo, z0) en hielo
# =============================================================================

CASOS = {
    "4400_DC": {"opt": (0.991, 0.0123), "medido": (0.990, 0.1557)},
    "4500_T" : {"opt": (0.954, 0.0203), "medido": (0.990, 0.0134)},
    "4500_DC": {"opt": (1.423, 0.0052), "medido": (1.268, 0.0559)},
    "4600_T" : {"opt": (0.978, 0.0618), "medido": (0.990, 0.0751)},
    "4500_I" : {"opt": (0.34, 0.0405),  "medido": (0.34, 0.0079)},
    "4600_I" : {"opt": (0.34, 0.0356),  "medido": (0.34, 0.0097)},
}

# =============================================================================
# CONSTANTES DEL MODELO
# =============================================================================

g     = CONSTANTS["g"];     k_vk  = CONSTANTS["k_vk"];  sigma = CONSTANTS["sigma"]
Rgas  = CONSTANTS["Rgas"];  Mair  = CONSTANTS["Mair"]
L_v   = CONSTANTS["L_v"];   L_f   = CONSTANTS["L_f"]
rho_w = CONSTANTS["rho_w"]; c_w   = CONSTANTS["c_w"];   c_ad  = CONSTANTS["c_ad"]
rho_i = CONSTANTS["rho_i"]; T_f   = CONSTANTS["T_f"]
timestep = MODEL["timestep"]; z_a = MODEL["z_a"]
range_   = MODEL["range_"];   N   = max(MODEL["N"], 10)

CRITICAL_VARS = ['t_a_k', 'rh_a_per', 'u_ms', 'sdown_wm2', 'ldown_wm2', 'p_a_pa']
GAP_WINDOW_DAYS, GAP_MIN_HOURS_VALID, GAP_SNOW_FRAC = 15, 20, 0.5

# =============================================================================
# SITIOS DESDE config.py (k/z0/albedo de config se ignoran: vienen de CASOS)
# =============================================================================

SITES_SWEEP = {}
for site, c in config.SITES.items():
    entry = {"meteo_file": c["meteo_file"], "d": float(c["d"])}
    if entry["d"] == 0.0:
        entry["epsilon_i"] = c["epsilon_i"]
    else:
        entry.update({k: c[k] for k in ("rho_d", "c_d", "epsilon_d", "albedo_d")})
    SITES_SWEEP[site] = entry

faltan = [s for s in CASOS if s not in SITES_SWEEP]
if faltan:
    raise KeyError(f"Sitios de CASOS comentados o ausentes en config.py: {faltan}")

# =============================================================================
# BALIZAS
# =============================================================================

stakes_df = pd.read_excel(STAKES_XLSX)
stakes_df['start_date'] = pd.to_datetime(stakes_df['start_date'])
stakes_df['end_date']   = pd.to_datetime(stakes_df['end_date'])
stakes_df['site_norm']  = stakes_df['site'].str.replace(r'_C$', '', regex=True)
stakes_df['lost'] = np.isclose(stakes_df['h_m'].astype(float), 3.0)


def balizas_sitio(site):
    s = stakes_df[stakes_df['site_norm'] == site].sort_values('end_date')
    if s.empty:
        return None
    return s[['start_date', 'end_date', 'cum_ablation_mweq',
              'cum_ablation_err_mweq', 'lost']].reset_index(drop=True)

# =============================================================================
# FORZANTE, SIMULACIÓN, ENVELOPE Y EVALUACIÓN (idénticos al barrido v2/v3)
# =============================================================================

def cargar_forzante(site, cfg, date_start, date_end):
    data = pd.read_csv(os.path.join(INPUT_DIR, cfg["meteo_file"]),
                       na_values=['NaN', 'NA', 'null', '', ' ', 'nan', 'N/A'])
    data['date'] = pd.to_datetime(data['date'])
    data = data[(data['date'] >= date_start) & (data['date'] <= date_end)].copy()

    cal = pd.date_range(pd.Timestamp(date_start).normalize(),
                        pd.Timestamp(date_end).normalize(), freq='D')

    if 'snow_presence' in data.columns:
        snow_daily = (data.set_index('date')['snow_presence'].fillna(0)
                      .resample('D').mean().reindex(cal).fillna(0))
    else:
        snow_daily = pd.Series(0.0, index=cal)
    snowy_day = (snow_daily >= GAP_SNOW_FRAC).values

    data = data.dropna(subset=CRITICAL_VARS).reset_index(drop=True)
    data['r_mm'] = data['r_mm'].fillna(0.0) if 'r_mm' in data else 0.0
    data['snow_presence'] = (data['snow_presence'].fillna(0).astype(int)
                             if 'snow_presence' in data else 0)
    data['rh_sfc_per'] = (data['rh_sfc_per'].clip(upper=100.0).fillna(100.0)
                          if 'rh_sfc_per' in data else 100.0)

    dates = data['date'].values
    T_a_C = data['t_a_k'].to_numpy(float) - 273.15
    p_a   = data['p_a_pa'].to_numpy(float)
    e_s   = 610.8 * np.exp(17.27 * T_a_C / (237.3 + T_a_C))
    e_a   = data['rh_a_per'].to_numpy(float) * e_s / 100.0
    q_a   = 0.622 * e_a / (p_a - 0.378 * e_a)
    dt_h  = np.diff(dates).astype('timedelta64[s]').astype(float) / 3600.0
    reset = np.concatenate([[False], dt_h > 3.0])

    hours_valid = (pd.Series(1.0, index=pd.DatetimeIndex(dates))
                   .resample('D').count().reindex(cal, fill_value=0))
    gap_day = (hours_valid < GAP_MIN_HOURS_VALID).values

    return {"dates": dates, "T_a_C": T_a_C,
            "Sdown": data['sdown_wm2'].to_numpy(float),
            "Ldown": data['ldown_wm2'].to_numpy(float),
            "u": data['u_ms'].to_numpy(float), "q_a": q_a,
            "RH_sfc": data['rh_sfc_per'].to_numpy(float),
            "r_ms": data['r_mm'].to_numpy(float) / (1000.0 * timestep),
            "p_a": p_a, "snow": data['snow_presence'].to_numpy(int),
            "reset": reset, "nt": len(data),
            "cal": cal, "cal_np": cal.values,
            "gap_day": gap_day, "snowy_day": snowy_day}


def simular_cum_debris(F, cfg, k_d, z_0_d):
    d = cfg["d"]
    T_s_in = float(F["T_a_C"][0])
    T_d_in = np.linspace(T_s_in, T_f, N, dtype=np.float64)[:-1]
    cum, cum_arr = 0.0, np.empty(F["nt"])
    for t in range(F["nt"]):
        Ta = float(F["T_a_C"][t])
        if F["reset"][t] or F["snow"][t] == 1:
            T_s_in = Ta
            T_d_in = np.linspace(Ta, T_f, N, dtype=np.float64)[:-1]
        if F["snow"][t] != 1:
            res = DEBmodel(
                timestep, float(F["Sdown"][t]), float(F["Ldown"][t]), Ta,
                float(F["u"][t]), float(F["q_a"][t]), float(F["RH_sfc"][t]),
                float(F["r_ms"][t]),
                T_s_in, T_d_in, d, z_a, z_0_d,
                k_d, cfg["rho_d"], cfg["c_d"], cfg["epsilon_d"],
                cfg["albedo_d"], range_, N,
                g, k_vk, sigma, Rgas, Mair, L_v, L_f,
                rho_w, c_w, c_ad, rho_i, T_f, float(F["p_a"][t]))
            cum   += float(res[0])
            T_s_in = float(res[1])
            T_d_in = np.array(res[2], dtype=np.float64)
        cum_arr[t] = cum
    return cum_arr


def simular_cum_ice(F, cfg, z_0_d, albedo_i):
    """Ablación de hielo = fusión + sublimación (idéntico al barrido v6)."""
    cum_m, cum_s = 0.0, 0.0
    cum_arr = np.empty(F["nt"])
    for t in range(F["nt"]):
        if F["snow"][t] != 1:
            res = CleanIceModel(
                timestep, float(F["Sdown"][t]), float(F["Ldown"][t]),
                float(F["T_a_C"][t]), float(F["u"][t]), float(F["q_a"][t]),
                float(F["r_ms"][t]),
                albedo_i, cfg["epsilon_i"], z_0_d,
                g, k_vk, sigma, Rgas, Mair, L_v, L_f,
                rho_w, c_w, c_ad, rho_i, T_f, float(F["p_a"][t]), z_a)
            cum_m += float(res[0])
            LE_t, T_s_t = float(res[6]), float(res[1])
            if LE_t < 0.0:
                L_eff = L_s if T_s_t < 0.0 else L_v
                cum_s += (-LE_t) * timestep / L_eff / 1000.0
        cum_arr[t] = cum_m + cum_s
    return cum_arr

def gap_envelope_run(F, cum_arr):
    melt_h = pd.Series(np.diff(cum_arr, prepend=0.0),
                       index=pd.DatetimeIndex(F["dates"]))
    daily = melt_h.resample('D').sum().reindex(F["cal"], fill_value=0.0)
    valid = daily.where(~F["gap_day"])
    typical = (valid.rolling(f'{2*GAP_WINDOW_DAYS + 1}D', center=True,
                             min_periods=5).mean().fillna(0.0).clip(lower=0.0))
    fill = np.where(F["gap_day"] & ~F["snowy_day"],
                    np.maximum(typical.values - daily.values, 0.0), 0.0)
    return daily.cumsum().values, (daily.values + fill).cumsum()


def evaluar(F, cum_min, cum_max, balizas):
    residuos, aceptables, detalle = [], [], []
    for _, b in balizas.iterrows():
        idx = np.searchsorted(F["cal_np"],
                              np.datetime64(b['end_date'].normalize()),
                              side='left')
        idx = min(idx, len(F["cal_np"]) - 1)
        s_min, s_max = cum_min[idx], cum_max[idx]
        s_mid = 0.5 * (s_min + s_max)
        obs, err = b['cum_ablation_mweq'], b['cum_ablation_err_mweq']
        lo, hi = obs - err, obs + err
        ok = (s_min <= hi) and (s_max >= lo)
        resid = s_mid - obs
        residuos.append(resid)
        aceptables.append(ok)
        detalle.append({"end_date": b['end_date'], "obs": obs, "err": err,
                        "lost": bool(b['lost']), "sim_min": s_min,
                        "sim_max": s_max, "sim_mid": s_mid,
                        "resid": resid, "ok": ok})
    rmse = float(np.sqrt(np.mean(np.array(residuos) ** 2)))
    return rmse, bool(np.all(aceptables)), detalle

# =============================================================================
# EXPERIMENTO
# =============================================================================

print(f"{'Sitio':9s} {'abl_opt':>8s} {'abl_med':>8s} {'Δ':>8s} {'Δ%':>7s} "
      f"{'RMSE_opt':>9s} {'RMSE_med':>9s}")
print("-" * 64)
resultados_caso = []
for site, casos in CASOS.items():
    cfg = SITES_SWEEP[site]
    balizas = balizas_sitio(site)
    F = cargar_forzante(site, cfg, balizas['start_date'].min(),
                        balizas['end_date'].max() + pd.Timedelta(days=1))
    es_hielo = (cfg["d"] == 0.0)
    fila = {"site": site}
    for caso, (p1, p2) in casos.items():
        if es_hielo:
            cum = simular_cum_ice(F, cfg, p2, p1)     # (z0, albedo)
        else:
            cum = simular_cum_debris(F, cfg, p1, p2)  # (k, z0)
        cum_min, cum_max = gap_envelope_run(F, cum)
        rmse, ok, det = evaluar(F, cum_min, cum_max, balizas)
        fila[f"abl_{caso}"]  = det[-1]['sim_mid']
        fila[f"rmse_{caso}"] = rmse
    dlt = fila['abl_medido'] - fila['abl_opt']
    dp  = 100 * dlt / fila['abl_opt']
    print(f"{site:9s} {fila['abl_opt']:8.3f} {fila['abl_medido']:8.3f} "
          f"{dlt:+8.3f} {dp:+6.1f}% {fila['rmse_opt']:9.3f} "
          f"{fila['rmse_medido']:9.3f}")
    resultados_caso.append({**fila, "delta": dlt, "delta_pct": dp})

df_out = pd.DataFrame(resultados_caso)
out_csv = os.path.join(OUT_DIR, "impacto_no_optimizar.csv")
df_out.to_csv(out_csv, index=False)
print(f"\n✅ Guardado en {out_csv}")