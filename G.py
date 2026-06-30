# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 12:51:55 2025

@author: ThinkPad
"""

def G(Ts, Td1, k_d, h):
    """
    Calcula el flujo de calor por conducción (G) en la capa superficial de detritos.

    Parámetros:
    - Ts: Temperatura superficial (°C)
    - Td1: Temperatura en la primera capa del detrito (°C)
    - k_d: Conductividad térmica del detrito (W/m·K)
    - h: Espesor de cada capa (m)

    Devuelve:
    - G: Flujo de calor conductivo (W/m²), positivo hacia abajo
    """
    return -k_d * (Ts - Td1) / h
