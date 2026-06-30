# -*- coding: utf-8 -*-

# ==========================================
# FUNCTION TO CALCULATE UPWELLING LONGWAVE HEAT FLUX (Lup)
# ==========================================
# Esta función calcula la radiación de onda larga emitida hacia arriba 
# desde una superficie usando la ley de Stefan-Boltzmann.

def Lup(T_sfc, epsilon, sigma):
    """
    Calcula el flujo de radiación de onda larga emitida por la superficie (W/m²).

    Parámetros:
    - T_sfc : temperatura de la superficie (°C)
    - epsilon : emisividad superficial (sin unidad)
    - sigma : constante de Stefan-Boltzmann (W/m²·K⁴)

    Retorna:
    - Lup : flujo de onda larga saliente (W/m², negativo por convención)
    """

    # Convertimos la temperatura de °C a K
    T_kelvin = T_sfc + 273.15

    # Aplicamos la ley de Stefan-Boltzmann: E = εσT⁴ (negativo = saliente)
    Lup = -epsilon * sigma * T_kelvin**4
    return Lup
