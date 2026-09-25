# -*- coding: utf-8 -*-
"""
CleanIceModel.py  —  v3
------------------------
Energy-balance model for bare ice and snow surfaces.

Follows the bare-ice scheme of Reid & Brock (2010, MATLAB original), with
the following documented deviations/extensions:

Changes vs v1 (MATLAB-faithful version)
---------------------------------------
FIX #A  Sub-freezing surface temperature
        The original pins T_s = T_f every timestep and clips negative melt
        to zero, so fluxes in non-melting hours are those of a fictitious
        0 °C surface. Here, when the total flux at T_f is negative, T_s is
        solved by bisection until the energy balance closes, clamped to
        T_COLD_MIN (default –40 °C). Melt is unchanged with respect to the
        original (both evaluate melt from the flux at T_f); what changes
        are the diagnosed fluxes, T_s and the LE-derived sublimation in
        cold hours, which the original misrepresents.

FIX #B  rho_w used in melt conversion
        The original divides by rho_i but labels the output m w.e.;
        converting energy to m w.e. requires rho_w. rho_i is kept in the
        signature for API compatibility but is not used in the melt line.

FIX #C  Extended return tuple — flux diagnostics
        Returns 9 values:
          (melt, T_s, Snet, Ldown, Lup_out, H_out, LE_out, P_out, totflux)
          index:  0    1     2      3        4       5       6      7      8

Changes vs v2
-------------
FIX #D  Bisection direction corrected (BUG)
        F(T_s) is monotonically DECREASING in T_s (a colder surface emits
        less and receives more turbulent heat): F(T_COLD_MIN) > 0 and
        F(T_f) < 0 in the cold branch. v2 used the update rule for an
        increasing function, so the solver never bracketed the root and
        ran to an endpoint: T_s ≈ –40 °C in windy hours (huge spurious H,
        deposition LE) or T_s ≈ 0 °C in calm hours (negative residual).
        Detected via the hourly energy-closure check (mean residuals of
        up to ~+1100 W m-2 in Feb–Apr 2024). Melt was NOT affected (melt
        branch does not use the solver); T_s, QH, QL, Lup and sublimation
        in cold hours were.

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

    (parámetros y retornos: idénticos a tu docstring actual — omitidos acá
     por brevedad; dejá el tuyo tal cual)
    """

    # ------------------------------------------------------------------
    # Shortwave — independent of T_s
    # ------------------------------------------------------------------
    Snet = Sdown * (1.0 - albedo_i)

    # Surface is assumed saturated (RH_sfc = 100 %)
    RH_sfc = 100.0

    def _eval_flux(T_s_try):
        """Return total flux and individual components for a given T_s."""
        Lup_   = Lup(T_s_try, epsilon_i, sigma)
        H_,  _ = H(T_a, T_s_try, u, q_a, z_a, z_0_i, g, c_ad, k_vk, Mair, Rgas, p_a)
        LE_    = Lat(T_a, T_s_try, u, q_a, RH_sfc, p_a, z_a, z_0_i, g, L_v, k_vk, Mair, Rgas)
        P_     = P(T_a, T_s_try, r, rho_w, c_w)
        tot_   = Snet + Ldown + Lup_ + H_ + LE_ + P_
        return tot_, Lup_, H_, LE_, P_

    # --- Step 1: evaluate at melting point (same trigger as the original) ---
    totflux_melt, Lup_out, H_out, LE_out, P_out = _eval_flux(T_f)

    if totflux_melt >= 0.0:
        # ---- MELTING CONDITIONS ----
        T_s     = T_f
        totflux = totflux_melt
        # FIX #B: divide by rho_w, not rho_i
        melt = totflux * timestep / (rho_w * L_f)

    else:
        # ---- SUB-FREEZING CONDITIONS: bisection between T_COLD_MIN and T_f
        # FIX #D — F(Ts) is DECREASING in Ts: F(T_lo = -40) > 0, F(T_hi = 0) < 0.
        # If F(mid) > 0 the root lies ABOVE mid → raise the floor (T_lo).
        # If no root exists above T_COLD_MIN (deep radiative deficit with
        # turbulence suppressed), T_hi walks down and T_s clamps at -40 °C
        # with a small negative residual — the intended clamp behaviour.
        T_lo = T_COLD_MIN
        T_hi = T_f

        for _ in range(_MAX_ITER):
            T_mid = 0.5 * (T_lo + T_hi)
            flux_mid, _, _, _, _ = _eval_flux(T_mid)
            if flux_mid > 0.0:
                T_lo = T_mid
            else:
                T_hi = T_mid
            if (T_hi - T_lo) < _TOL_C:
                break

        T_s = 0.5 * (T_lo + T_hi)

        # Re-evaluate fluxes at converged T_s
        totflux, Lup_out, H_out, LE_out, P_out = _eval_flux(T_s)

        # No melt when surface is sub-freezing
        melt = 0.0

    # ------------------------------------------------------------------
    # FIX #C — return extended tuple including T_s and totflux
    # ------------------------------------------------------------------
    return melt, T_s, Snet, Ldown, Lup_out, H_out, LE_out, P_out, totflux