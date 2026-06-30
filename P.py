# -*- coding: utf-8 -*-

# ===========================================
# FUNCTION TO CALCULATE HEAT FLUX FROM RAIN (P)
# ===========================================
# Esta función calcula el flujo de calor sensible proveniente de la lluvia,
# que depende de la masa de agua, su temperatura y su capacidad calorífica.

def P(T_a, T_s, rain, rho_w, c_w):
    """
    Calcula el flujo de calor aportado por la lluvia (W/m²).

    Parámetros:
    - T_a : temperatura del aire (K)
    - T_s : temperatura de la superficie (K)
    - rain : tasa de precipitación líquida (m/s)
    - rho_w : densidad del agua (kg/m³)
    - c_w : calor específico del agua (J/kg·K)

    Retorna:
    - P : flujo de calor por lluvia (W/m²)
    """

    # Flujo de calor = masa por calor específico por diferencia de temperatura
    # rain (m/s) × rho_w (kg/m³) = flujo másico (kg/m²/s)
    # Se multiplica por c_w × (T_a - T_s) para obtener flujo energético
    P = rho_w * rain * c_w * (T_a - T_s)
    return P
