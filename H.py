# -*- coding: utf-8 -*-

# ===============================================
# FUNCTION TO CALCULATE SENSIBLE HEAT FLUX (H)
# ===============================================
# Esta función estima el flujo de calor sensible (W/m²) desde el aire hacia la superficie,
# considerando gradientes térmicos, estabilidad atmosférica y humedad.

import numpy as np

def H(T_air, T_sfc, wind, q_air, z_a, z_0, g, c_ad, k_vk, Mair, Rgas, p_a):
    """
    Parámetros:
    - T_air: temperatura del aire (°C)
    - T_sfc: temperatura de la superficie (°C)
    - wind: velocidad del viento (m/s)
    - q_air: humedad específica del aire (kg/kg)
    - z_a: altura de medición del aire (m)
    - z_0: longitud de rugosidad (m)
    - g: aceleración de la gravedad (m/s²)
    - c_ad: calor específico del aire seco (J/kg·K)
    - k_vk: constante de von Karman (≈ 0.41)
    - Mair: masa molar del aire (kg/mol)
    - Rgas: constante universal de los gases (J/mol·K)
    - p_a: presión atmosférica (Pa)

    Retorna:
    - H: flujo de calor sensible (W/m²)
    - R_ib: número de Richardson (adimensional)
    """

    # Densidad del aire (kg/m³)
    T_air_K = T_air + 273.15
    rho_a = (p_a * Mair) / (Rgas * T_air_K)

    # Cálculo del número de Richardson (R_ib)
    delta_T = T_air - T_sfc
    T_sfc_K = T_sfc + 273.15
    T_mean_K = 0.5 * (T_air_K + T_sfc_K)

    if wind == 0:
        R_ib = 0
    else:
        R_ib = g * delta_T * (z_a - z_0) / (T_mean_K * wind**2)

    # Correcciones por estabilidad/instabilidad
    if R_ib > 0.2:
        R_ib = 0
    elif R_ib < -1:
        R_ib = 0

    # Calor específico del aire húmedo
    c_p = c_ad * (1 + 0.84 * q_air)

    # Función de estabilidad phi
    if R_ib >= 0:
        phi = (1 - 5 * R_ib)**2
    else:
        phi = (1 - 16 * R_ib)**0.75

    # Cálculo del flujo de calor sensible (W/m²)
    H = rho_a * c_p * (k_vk**2) * wind * delta_T * phi / (np.log(z_a / z_0)**2)

    return H, R_ib
