# -*- coding: utf-8 -*-
"""
debristemp.py

Function to calculate the debris temperature profile.
Original MATLAB code by Tim Reid, Copyright of The University of Dundee (2010).
Edited into function form for a distributed model (with Catriona Fyffe), 2011.
Translated to Python and verified for full equivalence (2025).

Inputs:
    Ts_t1    : float  - Surface temperature at new timestep, T_s(t+1) (°C)
    Ts_t     : float  - Surface temperature at previous timestep, T_s(t) (°C)
    Td       : array  - Debris temperature profile at previous timestep,
                        T_d(t,:), length N-1 (°C)
    N        : int    - Total number of layers (including surface)
    k_d      : float  - Debris thermal conductivity (W/m·K)
    timestep : float  - Time step (s)
    rho_d    : float  - Debris density (kg/m³)
    c_d      : float  - Debris specific heat capacity (J/kg·K)
    h        : float  - Thickness of each layer (m)
    T_f      : float  - Melting point temperature (°C)

Returns:
    debristemp : array - Updated debris temperature profile, length N-1 (°C)
"""

import numpy as np


def debristemp(Ts_t1, Ts_t, Td, N, k_d, timestep, rho_d, c_d, h, T_f):

    # -------------------------------------------------------------------------
    # Initialise output array (equivalent to MATLAB: debristemp = zeros(N-1,1))
    # -------------------------------------------------------------------------
    out = np.zeros(N - 1)

    # -------------------------------------------------------------------------
    # C, A and S are intermediate variables for the debris temp calculation.
    # Equivalent to MATLAB: C = k_d .* timestep ./ (2 .* rho_d .* c_d .* h.^2)
    # -------------------------------------------------------------------------
    C = k_d * timestep / (2.0 * rho_d * c_d * h**2)

    A = np.zeros(N - 1)
    S = np.zeros(N - 1)

    # -------------------------------------------------------------------------
    # Find value of A for top layer.
    # MATLAB: A(1) = 2*C + 1  →  Python: A[0] = 2*C + 1
    # -------------------------------------------------------------------------
    A[0] = 2.0 * C + 1.0

    # Find values of A for every other layer.
    # MATLAB: for y = 2:1:N-1  →  Python: range(1, N-1)  [indices 1 .. N-2]
    for y in range(1, N - 1):
        A[y] = 2.0 * C + 1.0 - (C**2 / A[y - 1])

    # -------------------------------------------------------------------------
    # Find value of S for top layer.
    # MATLAB: S(1) = C*Ts_t1 + C*Ts_t + (1-2*C)*Td(1) + C*Td(2)
    #         1-based Td(1) → Python Td[0];  Td(2) → Td[1]
    # -------------------------------------------------------------------------
    S[0] = C * Ts_t1 + C * Ts_t + (1.0 - 2.0 * C) * Td[0] + C * Td[1]

    # Find values of S for all the middle layers.
    # MATLAB: for y = 2:1:N-2  →  Python: range(1, N-2)  [indices 1 .. N-3]
    # MATLAB references: Td(y-1), Td(y), Td(y+1), S(y-1), A(y-1)
    # 0-based equivalents: Td[y-1], Td[y], Td[y+1], S[y-1], A[y-1]
    for y in range(1, N - 2):
        S[y] = (C * Td[y - 1]
                + (1.0 - 2.0 * C) * Td[y]
                + C * Td[y + 1]
                + C * S[y - 1] / A[y - 1])

    # Find value of S for bottom layer.
    # MATLAB: S(N-1) = 2*C*T_f + C*Td(N-2) + (1-2*C)*Td(N-1) + C*S(N-2)/A(N-2)
    # 1-based → 0-based:
    #   S(N-1)  → S[N-2]
    #   Td(N-2) → Td[N-3]
    #   Td(N-1) → Td[N-2]
    #   S(N-2)  → S[N-3]
    #   A(N-2)  → A[N-3]
    S[N - 2] = (2.0 * C * T_f
                + C * Td[N - 3]
                + (1.0 - 2.0 * C) * Td[N - 2]
                + C * S[N - 3] / A[N - 3])

    # -------------------------------------------------------------------------
    # Find new debris temperature in bottom layer.
    # MATLAB: debristemp(N-1) = S(N-1) / A(N-1)
    # 0-based: out[N-2] = S[N-2] / A[N-2]
    # -------------------------------------------------------------------------
    out[N - 2] = S[N - 2] / A[N - 2]

    # Find new debris temperature at all other layers (backsubstitution).
    # MATLAB: for y = 1:1:N-2
    #             debristemp(N-1-y) = (S(N-1-y) + C*debristemp(N-1-y+1)) / A(N-1-y)
    # When y=1 (MATLAB): debristemp(N-2) = (S(N-2) + C*debristemp(N-1)) / A(N-2)
    # 0-based equivalent: out[N-3] = (S[N-3] + C*out[N-2]) / A[N-3]
    # The loop works upward from the second-to-last layer to the top (index 0).
    # Iterating y from 1 to N-2 in MATLAB maps to index (N-2-y) in 0-based,
    # which runs from N-3 down to 0 — identical to range(N-3, -1, -1).
    for y in range(1, N - 1):
        out[N - 2 - y] = (S[N - 2 - y] + C * out[N - 1 - y]) / A[N - 2 - y]

    return out
