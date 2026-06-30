# -*- coding: utf-8 -*-

# ================================================
# SOLAR POSITION CALCULATION (Zenith and Azimuth)
# ================================================
# Equivalent to MATLAB's sun_calcs.m using sun_position() function.
# Uses pvlib with the NREL SPA algorithm for high-accuracy solar position.
# Location: Glacier site (lat=45.78, lon=6.88, alt=2065.78 m, UTC+1)
# Period: 2011-06-14 01:00 to 2011-09-13 00:00, 1-hour timesteps

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib

# ---------------------------
# 1. Site location
# ---------------------------
latitude  = 45.782698222
longitude = 6.877751313
altitude  = 2065.7845
tz = 'Europe/Paris'  # UTC+1, unambiguous (avoids POSIX sign inversion in Etc/GMT-*)

# ---------------------------
# 2. Time range (1-hour steps)
# ---------------------------
start = pd.Timestamp('2011-06-14 01:00:00', tz=tz)
end   = pd.Timestamp('2011-09-13 00:00:00', tz=tz)
times = pd.date_range(start=start, end=end, freq='1h')

# ---------------------------
# 3. Solar position calculation
# ---------------------------
# Using NREL SPA algorithm (Reda & Andreas, 2004) — high accuracy, 
# closest available equivalent to MATLAB's sun_position (Grena, 2008).
# Differences between both algorithms are typically < 0.01°.
location = pvlib.location.Location(latitude, longitude, tz=tz, altitude=altitude)
solpos   = location.get_solarposition(times, method='nrel_numpy')

# ---------------------------
# 4. Extract zenith and azimuth
# ---------------------------
Z = solpos['zenith'].values    # degrees
A = solpos['azimuth'].values   # degrees

# ---------------------------
# 5. Plot results
# ---------------------------
plt.figure(figsize=(12, 5))
plt.plot(Z, 'b', label='Zenith (°)')
plt.plot(A, 'r', label='Azimuth (°)')
plt.title('Solar position: Zenith and Azimuth — 2011')
plt.xlabel('Timestep (1h)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
