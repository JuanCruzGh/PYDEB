# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 13:06:29 2025

@author: ThinkPad
"""

# ==========================================
# DEGREE DAY MODEL (Modelo de Grado-Día)
# ==========================================

def DDM(d, T_a, t, Tcum):
    """
    Modelo de grado-día para fusión bajo detrito.
    Debe llamarse en cada timestep desde el loop principal.

    Parámetros:
    ----------
    d : float
        Espesor del detrito (m).
    T_a : float
        Temperatura del aire en el timestep actual (°C).
    t : int
        Timestep actual (en horas, desde el inicio de la simulación).
    Tcum : float
        Temperatura acumulada del día actual (°C), mantenida externamente.

    Retorna:
    -------
    Melt_DDM : float
        Fusión del día (m w.e.). Solo distinto de 0 al final de cada día.
    Tcum : float
        Temperatura acumulada actualizada para el siguiente timestep.
    """
    import numpy as np

    # Acumular temperatura horaria
    Tcum = Tcum + T_a

    Melt_DDM = 0  # Por defecto, sin fusión hasta fin de día

    # Solo calcular fusión al completar cada día (cada 24 horas)
    if t % 24 == 0:
        T_DDM = max(Tcum, 0)          # Solo temperaturas positivas
        DDF = 0.0027 * np.exp(-3.173 * d)  # m/°C/día
        Melt_DDM = DDF * T_DDM
        Tcum = 0  # Reiniciar acumulador diario

    return Melt_DDM, Tcum
