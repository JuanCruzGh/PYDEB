# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:47:43 2026

@author: ThinkPad
"""

# -*- coding: utf-8 -*-
"""
CleanIceModel.py  —  v2
------------------------
Energy-balance model for bare ice and snow surfaces.

Changes vs v1
-------------
FIX #A  Sub-freezing surface temperature
        T_s is no longer pinned at T_f = 0 °C every timestep.
        When the total energy flux is negative (surface cooling), T_s is
        solved iteratively until the energy balance closes, clamped to a
        minimum of T_COLD_MIN (default –40 °C).  Melt only occurs when the
        converged T_s reaches 0 °C AND totflux > 0.  This prevents spurious
        melt during cold periods (nights, winter) that was the main driver of
        overestimation in both clean-ice sites.

FIX #B  rho_w used in melt conversion
        The melt formula divides by rho_w (water density), not rho_i (ice
        density), because melt is expressed in m water equivalent.  The
        parameter rho_i received by this function is passed through but is
        NOT used in the melt line.  See NOTE below.

FIX #C  Extended return tuple — flux diagnostics
        The function now returns 9 values instead of 7:
          (melt, T_s, Snet, Ldown, Lup_out, H_out, LE_out, P_out, totflux)
          index:  0    1     2      3        4       5       6      7       8
        This lets the run script store per-timestep flux terms for diagnosing
        which component drives over/under-estimation.

NOTE: rho_i received here is the ICE density (kg/m³) from config.py.
      Melt (m w.e.) is computed with rho_w, not rho_i.  rho_i is accepted
      in the signature for API compatibility but is not used in melt calc.

Dependencies
------------
  Lup, H, Lat, P  — must be in the same directory.
"""

from Lup import Lup
from H   import H
from Lat import Lat
from P   import P

# Minimum physically plausible surface temperature (°C).
# Used to clamp the iterative T_s solver.
T_COLD_MIN = -40.0

# Maximum iterations and convergence tolerance for the T_s solver.
_MAX_ITER = 50
_TOL_C    = 0.01   # °C


def CleanIceModel(timestep, Sdown, Ldown, T_a, u, q_a, r,
                  albedo_i, epsilon_i, z_0_i,
                  g, k_vk, sigma, Rgas, Mair, L_v, L_f,
                  rho_w, c_w, c_ad, rho_i,
                  T_f, p_a, z_a):
    """
    Energy-balance model for a bare-ice or snow surface.

    Parameters
    ----------
    timestep  : int    time step (s)
    Sdown     : float  downwelling shortwave radiation (W/m²)
    Ldown     : float  downwelling longwave radiation (W/m²)
    T_a       : float  air temperature (°C)
    u         : float  wind speed (m/s)
    q_a       : float  specific humidity of air (kg/kg)
    r         : float  rainfall rate (m/s)
    albedo_i  : float  surface albedo
    epsilon_i : float  surface emissivity
    z_0_i     : float  aerodynamic roughness length (m)
    g         : float  gravitational acceleration (m/s²)
    k_vk      : float  von Kármán constant
    sigma     : float  Stefan-Boltzmann constant (W/m²·K⁴)
    Rgas      : float  universal gas constant (J/mol·K)
    Mair      : float  molar mass of dry air (kg/mol)
    L_v       : float  latent heat of vaporisation (J/kg)
    L_f       : float  latent heat of fusion (J/kg)
    rho_w     : float  density of water (kg/m³)  — used in melt conversion
    c_w       : float  specific heat of water (J/kg·K)
    c_ad      : float  specific heat of dry air (J/kg·K)
    rho_i     : float  density of ice (kg/m³)  — NOT used in melt calc (see NOTE)
    T_f       : float  melting point (°C), normally 0.0
    p_a       : float  atmospheric pressure (Pa)
    z_a       : float  measurement height (m)

    Returns  (index)
    ----------------
    0  melt      : float  melt rate (m w.e. per timestep); ≥ 0
    1  T_s       : float  converged surface temperature (°C)
    2  Snet      : float  net shortwave radiation (W/m²)
    3  Ldown     : float  downwelling longwave (W/m²)  [pass-through]
    4  Lup_out   : float  upwelling longwave (W/m²)
    5  H_out     : float  sensible heat flux (W/m²)
    6  LE_out    : float  latent heat flux (W/m²)
    7  P_out     : float  precipitation heat flux (W/m²)
    8  totflux   : float  total energy flux at converged T_s (W/m²)
    """

    # ------------------------------------------------------------------
    # Shortwave — independent of T_s
    # ------------------------------------------------------------------
    Snet = Sdown * (1.0 - albedo_i)

    # Surface is assumed saturated (RH_sfc = 100 %)
    RH_sfc = 100.0

    # ------------------------------------------------------------------
    # FIX #A — iterative T_s solver
    #
    # Strategy:
    #   1. First evaluate total flux assuming T_s = T_f (melting point).
    #   2. If totflux >= 0  →  surface is at or above melting; T_s = T_f,
    #      melt = totflux * timestep / (rho_w * L_f).
    #   3. If totflux < 0   →  surface cools below 0 °C.  Iterate T_s
    #      downward until the residual flux closes (≈ 0), meaning the
    #      longwave and turbulent terms re-balance the shortwave deficit.
    #      No melt occurs in this case.
    # ------------------------------------------------------------------

    def _eval_flux(T_s_try):
        """Return total flux and individual components for a given T_s."""
        Lup_   = Lup(T_s_try, epsilon_i, sigma)
        H_,  _ = H(T_a, T_s_try, u, q_a, z_a, z_0_i, g, c_ad, k_vk, Mair, Rgas, p_a)
        LE_    = Lat(T_a, T_s_try, u, q_a, RH_sfc, p_a, z_a, z_0_i, g, L_v, k_vk, Mair, Rgas)
        P_     = P(T_a, T_s_try, r, rho_w, c_w)
        tot_   = Snet + Ldown + Lup_ + H_ + LE_ + P_
        return tot_, Lup_, H_, LE_, P_

    # --- Step 1: evaluate at melting point ---
    totflux_melt, Lup_out, H_out, LE_out, P_out = _eval_flux(T_f)

    if totflux_melt >= 0.0:
        # ---- MELTING CONDITIONS ----
        T_s     = T_f
        totflux = totflux_melt
        # FIX #B: divide by rho_w, not rho_i
        melt = totflux * timestep / (rho_w * L_f)
        if melt < 0.0:
            melt = 0.0

    else:
        # ---- SUB-FREEZING CONDITIONS: iterate T_s ----
        # Simple bisection between T_f and T_COLD_MIN.
        T_lo = T_COLD_MIN
        T_hi = T_f

        for _ in range(_MAX_ITER):
            T_mid = 0.5 * (T_lo + T_hi)
            flux_mid, _, _, _, _ = _eval_flux(T_mid)
            if flux_mid > 0.0:
                T_hi = T_mid
            else:
                T_lo = T_mid
            if (T_hi - T_lo) < _TOL_C:
                break

        T_s = 0.5 * (T_lo + T_hi)

        # Clamp to physically meaningful range
        if T_s > T_f:
            T_s = T_f
        if T_s < T_COLD_MIN:
            T_s = T_COLD_MIN

        # Re-evaluate fluxes at converged T_s
        totflux, Lup_out, H_out, LE_out, P_out = _eval_flux(T_s)

        # No melt when surface is sub-freezing
        melt = 0.0

    # ------------------------------------------------------------------
    # FIX #C — return extended tuple including T_s and totflux
    # ------------------------------------------------------------------
    return melt, T_s, Snet, Ldown, Lup_out, H_out, LE_out, P_out, totflux