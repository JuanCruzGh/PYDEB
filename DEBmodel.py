# -*- coding: utf-8 -*-
"""
DEBmodel.py

Energy-balance model for calculating glacier melt rates under a debris layer.
Original MATLAB code by Tim Reid, Copyright of The University of Dundee (2010).
Edited into function form for a distributed model (with Catriona Fyffe), 2011.
Translated to Python and corrected for full equivalence (2025).

Dependencies: debristemp.py, Lup.py, G.py, H.py, Lat.py, P.py
All must be saved in the same directory as this file.

NOTE: rho_i as taken in by the model will actually have the rho_w value, as
you should use water density not ice density to convert energy to melt in m w.e.
"""

import numpy as np
from debristemp import debristemp
from Lup import Lup
from G import G
from H import H
from Lat import Lat
from P import P


def DEBmodel(timestep, Sdown, Ldown, T_a, u, q_a, RH_sfc, r,
             T_s_in, T_d_in, d, z_a, z_0_d, k_d, rho_d, c_d,
             epsilon_d, albedo_d, range_, N,
             g, k_vk, sigma, Rgas, Mair, L_v, L_f,
             rho_w, c_w, c_ad, rho_i, T_f, p_a):
    """
    Energy-balance model for glacier melt under a debris layer.

    Parameters
    ----------
    timestep  : float  - time step in seconds
    Sdown     : float  - downwelling shortwave radiation (W/m²)
    Ldown     : float  - downwelling longwave radiation (W/m²)
    T_a       : float  - air temperature (°C)
    u         : float  - wind speed (m/s)
    q_a       : float  - specific humidity of air (kg/kg)
    RH_sfc    : float  - surface relative humidity (%)
    r         : float  - rainfall rate (m/s)
    T_s_in    : float  - surface temperature at previous timestep (°C)
    T_d_in    : array  - debris internal temperature profile, length N-1 (°C)
    d         : float  - debris thickness (m)
    z_a       : float  - measurement height (m)
    z_0_d     : float  - debris roughness length (m)
    k_d       : float  - debris thermal conductivity (W/m·K)
    rho_d     : float  - debris density (kg/m³)
    c_d       : float  - debris specific heat capacity (J/kg·K)
    epsilon_d : float  - debris emissivity
    albedo_d  : float  - debris albedo
    range_    : float  - perturbation step for numerical derivative (°C)
    N         : int    - number of debris layers
    g         : float  - gravitational acceleration (m/s²)
    k_vk      : float  - von Kármán constant
    sigma     : float  - Stefan-Boltzmann constant (W/m²·K⁴)
    Rgas      : float  - universal gas constant (J/mol·K)
    Mair      : float  - molar mass of dry air (kg/mol)
    L_v       : float  - latent heat of vaporisation (J/kg)
    L_f       : float  - latent heat of fusion (J/kg)
    rho_w     : float  - density of water (kg/m³)
    c_w       : float  - specific heat of water (J/kg·K)
    c_ad      : float  - specific heat of air (J/kg·K)
    rho_i     : float  - density used for melt conversion (pass rho_w, see note above)
    T_f       : float  - melting point of water (°C)
    p_a       : float  - atmospheric pressure (Pa)

    Returns
    -------
    melt      : float  - melt rate (m water equivalent)
    T_s_out   : float  - output surface temperature (°C)
    T_d_out   : array  - output debris internal temperature profile (°C)
    Snet      : float  - net shortwave radiation (W/m²)
    Lup_out   : float  - upwelling longwave radiation (W/m²)
    G_out     : float  - conductive heat flux at surface (W/m²)
    H_out     : float  - sensible heat flux (W/m²)
    LE_out    : float  - latent heat flux (W/m²)
    P_out     : float  - rainfall heat flux (W/m²)
    """

    # -------------------------------------------------------------------------
    # Guard check carried over from MATLAB (incomplete in original — placeholder)
    # -------------------------------------------------------------------------
    if T_s_in < -237.3:
        pass  # Guard check from original MATLAB (no action defined — add handling if needed)

    # -------------------------------------------------------------------------
    # Layer thickness
    # -------------------------------------------------------------------------
    h = d / N  # Divide debris thickness by N (typically 10)

    # -------------------------------------------------------------------------
    # Net shortwave radiation
    # -------------------------------------------------------------------------
    Snet = Sdown * (1 - albedo_d)

    # =========================================================================
    # MAIN MODEL LOOP — Newton-Raphson iteration for surface temperature
    # =========================================================================
    # MATLAB uses 1-based indexing: Ts(1)..Ts(102)
    # Here we use a 102-element array (indices 0..101) but populate from index 1
    # to mirror MATLAB's Ts(1)=T_s_in, Ts(2)=T_s_in-0.5, n starting at 2.

    Ts = np.zeros(102)
    Ts[1] = T_s_in          # MATLAB: Ts(n-1) = T_s_in  with n=2 → Ts(1)
    Ts[2] = T_s_in - 0.5   # MATLAB: Ts(n)   = T_s_in - 0.5
    n = 2

    while abs(Ts[n] - Ts[n - 1]) > 0.01 and n < 100:

        # Debris temperature profiles for current and perturbed surface temps
        Td       = debristemp(Ts[n],          T_s_in, T_d_in, N, k_d, timestep, rho_d, c_d, h, T_f)
        Td_plus  = debristemp(Ts[n] + range_, T_s_in, T_d_in, N, k_d, timestep, rho_d, c_d, h, T_f)
        Td_minus = debristemp(Ts[n] - range_, T_s_in, T_d_in, N, k_d, timestep, rho_d, c_d, h, T_f)

        # -----------------------------------------------------------------
        # Newton-Raphson numerator: f(Ts)
        # -----------------------------------------------------------------
        num = (Snet
               + Ldown
               + Lup(Ts[n],          epsilon_d, sigma)
               + G(Ts[n],          Td[0],       k_d, h)
               + H(T_a, Ts[n],          u, q_a, z_a, z_0_d, g, c_ad, k_vk, Mair, Rgas, p_a)[0]
               + Lat(T_a, Ts[n],          u, q_a, RH_sfc, p_a, z_a, z_0_d, g, L_v, k_vk, Mair, Rgas)
               + P(T_a, Ts[n],          r, rho_w, c_w))

        # -----------------------------------------------------------------
        # Newton-Raphson denominator: [f(Ts+range) - f(Ts-range)] / (2*range)
        # i.e. numerical derivative of the energy balance w.r.t. Ts
        # -----------------------------------------------------------------
        f_plus = (Snet
                  + Ldown
                  + Lup(Ts[n] + range_, epsilon_d, sigma)
                  + G(Ts[n] + range_, Td_plus[0],  k_d, h)
                  + H(T_a, Ts[n] + range_, u, q_a, z_a, z_0_d, g, c_ad, k_vk, Mair, Rgas, p_a)[0]
                  + Lat(T_a, Ts[n] + range_, u, q_a, RH_sfc, p_a, z_a, z_0_d, g, L_v, k_vk, Mair, Rgas)
                  + P(T_a, Ts[n] + range_, r, rho_w, c_w))

        f_minus = (Snet
                   + Ldown
                   + Lup(Ts[n] - range_, epsilon_d, sigma)
                   + G(Ts[n] - range_, Td_minus[0], k_d, h)
                   + H(T_a, Ts[n] - range_, u, q_a, z_a, z_0_d, g, c_ad, k_vk, Mair, Rgas, p_a)[0]
                   + Lat(T_a, Ts[n] - range_, u, q_a, RH_sfc, p_a, z_a, z_0_d, g, L_v, k_vk, Mair, Rgas)
                   + P(T_a, Ts[n] - range_, r, rho_w, c_w))

        den = (f_plus - f_minus) / (2 * range_)

        # Newton-Raphson update
        Ts[n + 1] = Ts[n] - num / den

        # -----------------------------------------------------------------
        # Step limiter: no step larger than 1 °C (protects against
        # low-derivative instability). Two separate if-blocks as in MATLAB.
        # NOTE: np.clip was NOT used here — it clips the absolute value of
        # Ts[n+1], not the step size, which would be incorrect.
        # -----------------------------------------------------------------
        if Ts[n + 1] - Ts[n] > 1:
            Ts[n + 1] = Ts[n] + 1
        if Ts[n + 1] - Ts[n] < -1:
            Ts[n + 1] = Ts[n] - 1


        n += 1

    # =========================================================================
    # END OF MAIN MODEL LOOP
    # =========================================================================

    # Determine output surface temperature
    # MATLAB: if n==100, average Ts(99) and Ts(100) [1-based] → indices 98 and 99 [0-based]
    if n == 100:
        T_s_out = (Ts[98] + Ts[99]) / 2
    else:
        T_s_out = Ts[n]

    # -------------------------------------------------------------------------
    # Calculate debris internal temperature profile for final surface temperature
    # -------------------------------------------------------------------------
    T_d_out = debristemp(T_s_out, T_s_in, T_d_in, N, k_d, timestep, rho_d, c_d, h, T_f)

    # =========================================================================
    # CALCULATE MELT RATE
    # =========================================================================
    # Conductive flux G_i at the debris base (into the ice).
    # MATLAB: T_d_out(N-1) with 1-based indexing on an array of length N-1
    # → last element → Python index [N-2]
    G_i = k_d * (T_d_out[N - 2] - T_f) / h   # W/m²

    melt = G_i * timestep / (rho_w * L_f)     # m water equivalent

    if melt < 0:
        melt = 0

    # =========================================================================
    # RECALCULATE SURFACE FLUXES AT FINAL SURFACE TEMPERATURE
    # =========================================================================
    Lup_out = Lup(T_s_out, epsilon_d, sigma)
    G_out   = G(T_s_out, T_d_out[0], k_d, h)
    H_out, R_ib   = H(T_a, T_s_out, u, q_a, z_a, z_0_d, g, c_ad, k_vk, Mair, Rgas, p_a)
    Lat_out = Lat(T_a, T_s_out, u, q_a, RH_sfc, p_a, z_a, z_0_d, g, L_v, k_vk, Mair, Rgas)
    P_out   = P(T_a, T_s_out, r, rho_w, c_w)
    LE_out  = Lat_out  # LE_out and Lat_out are both latent heat (as in MATLAB)

    return melt, T_s_out, T_d_out, Snet, Lup_out, G_out, H_out, LE_out, P_out
