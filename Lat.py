# -*- coding: utf-8 -*-

# ================================================
# FUNCTION TO CALCULATE LATENT HEAT FLUX (Lat)
# ================================================

import numpy as np

def Lat(T_air, T_sfc, wind, q_air, RH_sfc, p_a, z_a, z_0, g, L_v, k_vk, Mair, Rgas):
    """
    Calcula el flujo de calor latente (W/m²) desde la superficie, 
    en función de las condiciones atmosféricas y estabilidad.

    Parámetros:
    - T_air: temperatura del aire (°C)
    - T_sfc: temperatura de la superficie (°C)
    - wind: velocidad del viento (m/s)
    - q_air: humedad específica del aire (kg/kg)
    - RH_sfc: humedad relativa de la superficie (%)
    - p_a: presión atmosférica (Pa)
    - z_a: altura de medición (m)
    - z_0: longitud de rugosidad (m)
    - g: gravedad (m/s²)
    - L_v: calor latente de vaporización (J/kg)
    - k_vk: constante de von Kármán (~0.41)
    - Mair: masa molar del aire (kg/mol)
    - Rgas: constante universal de los gases (J/mol·K)

    Retorna:
    - Lat: flujo de calor latente (W/m²)
    """

    # Si la humedad relativa superficial no es saturada, no hay evaporación/sublimación
    if RH_sfc < 100:
        return 0

    # Conversión a Kelvin
    T_air_K = T_air + 273.15
    T_sfc_K = T_sfc + 273.15

    # Densidad del aire (kg/m³)
    rho_a = (p_a * Mair) / (Rgas * T_air_K)

    # Presión de vapor de saturación en la superficie (Pa)
    es_s = 610.8 * np.exp((17.27 * T_sfc) / (237.3 + T_sfc))

    # Presión de vapor real en superficie
    es = RH_sfc * es_s / 100

    # Humedad específica en la superficie
    q_sfc = 0.622 * es / (p_a - 0.378 * es)

    # Número de Richardson (estabilidad atmosférica)
    if wind == 0:
        R_ib = 0
    else:
        T_mean_K = 0.5 * (T_air_K + T_sfc_K)
        R_ib = g * (T_air - T_sfc) * (z_a - z_0) / (T_mean_K * wind**2)

    # Corrección por estabilidad alta (flujo laminar cesa por encima de 0.2)
    if R_ib > 0.2:
        R_ib = 0

    # Corrección por inestabilidad extrema (solo convección libre por debajo de -1)
    if R_ib < -1:
        R_ib = 0

    # Función de estabilidad (phi)
    if R_ib >= 0:
        phi = (1 - 5 * R_ib)**2
    else:
        phi = (1 - 16 * R_ib)**0.75

    # Cálculo del flujo latente (W/m²)
    Lat = (rho_a * L_v * (k_vk**2) * wind * (q_air - q_sfc) * phi) / (np.log(z_a / z_0)**2)

    return Lat
