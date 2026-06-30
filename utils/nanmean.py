# -*- coding: utf-8 -*-

# ================================================
# FUNCTION TO CALCULATE MEAN IGNORING NaNs
# ================================================
# Equivalent to MATLAB's nanmean: when no dimension is specified,
# operates column-wise (axis=0), matching MATLAB's default sum behavior.

import numpy as np

def nanmean(x, dim=None):
    """
    Calcula la media ignorando los valores NaN, como en MATLAB.

    Parámetros:
    ----------
    x : ndarray
        Array de entrada, puede contener valores np.nan.
    dim : int, optional
        Dimensión a lo largo de la cual calcular la media.
        Si no se especifica, opera a lo largo del eje 0 (columnas),
        igual que MATLAB cuando se llama con un solo argumento.

    Retorna:
    -------
    nanmean : ndarray o float
        Media de `x` ignorando los NaN a lo largo del eje especificado.
    """

    x = np.array(x, dtype=float)  # Asegurar que sea tipo float para que acepte NaNs

    # Crear máscara de NaNs
    nans = np.isnan(x)

    # Reemplazar NaNs por cero para poder sumar sin error
    x_no_nan = np.where(nans, 0, x)

    if dim is None:
        # Operar a lo largo del eje 0 (column-wise), igual que MATLAB sum(x) por defecto
        count = np.sum(~nans, axis=0).astype(float)
        count[count == 0] = np.nan  # Evitar división por cero
        return np.sum(x_no_nan, axis=0) / count
    else:
        # Contar valores válidos a lo largo de la dimensión especificada
        count = np.sum(~nans, axis=dim).astype(float)
        count[count == 0] = np.nan  # Evitar división por cero

        # Sumar los valores válidos
        total = np.sum(x_no_nan, axis=dim)

        return total / count
