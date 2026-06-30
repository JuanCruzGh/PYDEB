# -*- coding: utf-8 -*-

# =============================================================================
# FUNCTION TO CALCULATE SUN POSITION (Zenith and Azimuth)
# =============================================================================
# Implementation of the algorithm presented by Reda & Andreas (2003):
#   "Solar position algorithm for solar radiation applications"
#   NREL Technical Report NREL/TP-560-34302
#   Available at: www.osti.gov/bridge
#
# Accuracy: +/- 0.0003 degrees (per original authors).
# Atmospheric refraction correction assumes T=283 K and P=1010 mbar.
#
# Input parameters:
#   time : dict with keys:
#       'year'  : year (valid range: -2000 to 6000)
#       'month' : month [1-12]
#       'day'   : calendar day [1-31]
#       'hour'  : local hour [0-23]
#       'min'   : minute [0-59]
#       'sec'   : second [0-59]
#       'UTC'   : offset from UTC in hours (local time = UTC time + UTC offset)
#   location : dict with keys:
#       'latitude'  : degrees north of equator (positive north)
#       'longitude' : degrees east of Greenwich (positive east)
#       'altitude'  : metres above mean sea level
#
# Output:
#   sun : dict with keys:
#       'zenith'  : zenith angle in degrees (angle from vertical)
#       'azimuth' : azimuth angle in degrees, eastward from north
#
# Example:
#   location = {'longitude': -105.1786, 'latitude': 39.742476, 'altitude': 1830.14}
#   time     = {'year': 2003, 'month': 10, 'day': 17,
#                'hour': 12, 'min': 30, 'sec': 30, 'UTC': -7}
#   sun = sun_position(time, location)
#   # Expected: zenith ≈ 50.1080, azimuth ≈ 194.3412
#
# History (original MATLAB):
#   09/03/2004  Original creation — Vincent Roy (vincent.roy@drdc-rddc.gc.ca)
#   10/03/2004  Fixed Julian calculation bug for year 1582 — Vincent Roy
#   18/03/2004  Header correction only — Vincent Roy
#   13/04/2004  Accept MATLAB date string input — Vincent Roy
#   22/08/2005  Julian routine compliance update — Vincent Roy
# Python translation: faithful line-by-line port of the above MATLAB code.
# =============================================================================

import numpy as np


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def sun_position(time, location):
    """
    Compute sun zenith and azimuth angles for given time and location.

    Parameters
    ----------
    time : dict
        Keys: 'year', 'month', 'day', 'hour', 'min', 'sec', 'UTC'
    location : dict
        Keys: 'latitude', 'longitude', 'altitude'

    Returns
    -------
    sun : dict
        Keys: 'zenith' (degrees from vertical), 'azimuth' (degrees, N=0, E=90)
    """

    # Step 1: Julian day, century, ephemeris day, century, millennium
    julian = _julian_calculation(time)

    # Step 2: Earth heliocentric longitude, latitude, radius (L, B, R)
    earth_helio = _earth_heliocentric_position_calculation(julian)

    # Step 3: Sun geocentric longitude and latitude
    sun_geo = _sun_geocentric_position_calculation(earth_helio)

    # Step 4: Nutation in longitude and obliquity (degrees)
    nutation = _nutation_calculation(julian)

    # Step 5: True obliquity of the ecliptic (degrees)
    true_obliquity = _true_obliquity_calculation(julian, nutation)

    # Step 6: Aberration correction (degrees)
    aberration_correction = _aberration_correction_calculation(earth_helio)

    # Step 7: Apparent sun longitude (degrees)
    apparent_sun_longitude = _apparent_sun_longitude_calculation(
        sun_geo, nutation, aberration_correction)

    # Step 8: Apparent sidereal time at Greenwich (degrees)
    apparent_stime = _apparent_stime_at_greenwich_calculation(
        julian, nutation, true_obliquity)

    # Step 9: Sun right ascension (degrees)
    sun_ra = _sun_right_ascension_calculation(
        apparent_sun_longitude, true_obliquity, sun_geo)

    # Step 10: Sun geocentric declination (degrees)
    sun_dec = _sun_geocentric_declination_calculation(
        apparent_sun_longitude, true_obliquity, sun_geo)

    # Step 11: Observer local hour angle (degrees, westward from south)
    observer_local_hour = _observer_local_hour_calculation(
        apparent_stime, location, sun_ra)

    # Step 12: Topocentric sun position (RA, declination, RA parallax) in degrees
    topo_sun = _topocentric_sun_position_calculate(
        earth_helio, location, observer_local_hour, sun_ra, sun_dec)

    # Step 13: Topocentric local hour angle (degrees)
    topo_local_hour = _topocentric_local_hour_calculate(
        observer_local_hour, topo_sun)

    # Step 14: Topocentric zenith and azimuth angles (degrees)
    sun = _sun_topocentric_zenith_angle_calculate(
        location, topo_sun, topo_local_hour)

    return sun


# =============================================================================
# SUBFUNCTIONS
# =============================================================================

def _set_to_range(var, min_interval, max_interval):
    """
    Wrap var into [min_interval, max_interval).
    Direct translation of MATLAB set_to_range subfunction.
    """
    var = var - max_interval * np.floor(var / max_interval)
    if var < min_interval:
        var = var + max_interval
    return var


def _julian_calculation(t_input):
    """
    Compute Julian day, century, ephemeris day, ephemeris century,
    and ephemeris millennium from local time and UTC offset.
    delta_t = 0 (as in the original MATLAB code).
    """
    # If input is not a dict, treat as a date string (UTC=0)
    if not isinstance(t_input, dict):
        import datetime
        dt = t_input  # assume datetime-like or parseable string
        if isinstance(dt, str):
            from dateutil import parser as dateparser
            dt = dateparser.parse(dt)
        time = {
            'year': dt.year, 'month': dt.month, 'day': dt.day,
            'hour': dt.hour, 'min': dt.minute, 'sec': dt.second,
            'UTC': 0
        }
    else:
        time = t_input

    year  = time['year']
    month = time['month']
    day   = time['day']
    hour  = time['hour']
    min_  = time['min']
    sec   = time['sec']
    UTC   = time['UTC']

    # Adjust year/month for January and February
    if month in (1, 2):
        Y = year - 1
        M = month + 12
    else:
        Y = year
        M = month

    # UT time as fractional day
    ut_time = ((hour - UTC) / 24.0) + (min_ / (60.0 * 24.0)) + (sec / (3600.0 * 24.0))
    D = day + ut_time  # decimal day of month

    # Gregorian calendar correction (adopted in 1582)
    if year == 1582:
        if month == 10:
            if day <= 4:          # Julian calendar ended Oct 4 1582
                B = 0
            elif day >= 15:       # Gregorian calendar started Oct 15 1582
                A = np.floor(Y / 100.0)
                B = 2 - A + np.floor(A / 4.0)
            else:
                # Dates Oct 5–14 1582 never existed
                print('This date never existed! Date automatically set to October 4, 1582')
                month = 10
                day   = 4
                B     = 0
        elif month < 10:          # Julian calendar
            B = 0
        else:                     # Gregorian calendar
            A = np.floor(Y / 100.0)
            B = 2 - A + np.floor(A / 4.0)
    elif year < 1582:             # Julian calendar
        B = 0
    else:                         # Gregorian calendar
        A = np.floor(Y / 100.0)
        B = 2 - A + np.floor(A / 4.0)

    JD  = np.floor(365.25 * (Y + 4716)) + np.floor(30.6001 * (M + 1)) + D + B - 1524.5

    delta_t = 0  # seconds (33.184 in some versions; set to 0 as in MATLAB code)
    JDE = JD + (delta_t / 86400.0)

    JC  = (JD  - 2451545.0) / 36525.0
    JCE = (JDE - 2451545.0) / 36525.0
    JME = JCE / 10.0

    return {
        'day':                  JD,
        'century':              JC,
        'ephemeris_day':        JDE,
        'ephemeris_century':    JCE,
        'ephemeris_millenium':  JME
    }


def _earth_heliocentric_position_calculation(julian):
    """
    Compute Earth heliocentric longitude (degrees), latitude (degrees),
    and radius vector (AU) using tabulated VSOP87 series coefficients.
    Each row is [A, B, C]; term = A * cos(B + C*JME).
    """
    JME = julian['ephemeris_millenium']

    # ------------------------------------------------------------------
    # LONGITUDE terms (L0 … L5)  [A, B, C]
    # ------------------------------------------------------------------
    L0_terms = np.array([
        [175347046.0, 0.0,        0.0       ],
        [3341656.0,   4.6692568,  6283.07585],
        [34894.0,     4.6261,     12566.1517],
        [3497.0,      2.7441,     5753.3849 ],
        [3418.0,      2.8289,     3.5231    ],
        [3136.0,      3.6277,     77713.7715],
        [2676.0,      4.4181,     7860.4194 ],
        [2343.0,      6.1352,     3930.2097 ],
        [1324.0,      0.7425,     11506.7698],
        [1273.0,      2.0371,     529.691   ],
        [1199.0,      1.1096,     1577.3435 ],
        [990.0,       5.233,      5884.927  ],
        [902.0,       2.045,      26.298    ],
        [857.0,       3.508,      398.149   ],
        [780.0,       1.179,      5223.694  ],
        [753.0,       2.533,      5507.553  ],
        [505.0,       4.583,      18849.228 ],
        [492.0,       4.205,      775.523   ],
        [357.0,       2.92,       0.067     ],
        [317.0,       5.849,      11790.629 ],
        [284.0,       1.899,      796.298   ],
        [271.0,       0.315,      10977.079 ],
        [243.0,       0.345,      5486.778  ],
        [206.0,       4.806,      2544.314  ],
        [205.0,       1.869,      5573.143  ],
        [202.0,       2.4458,     6069.777  ],
        [156.0,       0.833,      213.299   ],
        [132.0,       3.411,      2942.463  ],
        [126.0,       1.083,      20.775    ],
        [115.0,       0.645,      0.98      ],
        [103.0,       0.636,      4694.003  ],
        [102.0,       0.976,      15720.839 ],
        [102.0,       4.267,      7.114     ],
        [99.0,        6.21,       2146.17   ],
        [98.0,        0.68,       155.42    ],
        [86.0,        5.98,       161000.69 ],
        [85.0,        1.3,        6275.96   ],
        [85.0,        3.67,       71430.7   ],
        [80.0,        1.81,       17260.15  ],
        [79.0,        3.04,       12036.46  ],
        [71.0,        1.76,       5088.63   ],
        [74.0,        3.5,        3154.69   ],
        [74.0,        4.68,       801.82    ],
        [70.0,        0.83,       9437.76   ],
        [62.0,        3.98,       8827.39   ],
        [61.0,        1.82,       7084.9    ],
        [57.0,        2.78,       6286.6    ],
        [56.0,        4.39,       14143.5   ],
        [56.0,        3.47,       6279.55   ],
        [52.0,        0.19,       12139.55  ],
        [52.0,        1.33,       1748.02   ],
        [51.0,        0.28,       5856.48   ],
        [49.0,        0.49,       1194.45   ],
        [41.0,        5.37,       8429.24   ],
        [41.0,        2.4,        19651.05  ],
        [39.0,        6.17,       10447.39  ],
        [37.0,        6.04,       10213.29  ],
        [37.0,        2.57,       1059.38   ],
        [36.0,        1.71,       2352.87   ],
        [36.0,        1.78,       6812.77   ],
        [33.0,        0.59,       17789.85  ],
        [30.0,        0.44,       83996.85  ],
        [30.0,        2.74,       1349.87   ],
        [25.0,        3.16,       4690.48   ],
    ])

    L1_terms = np.array([
        [628331966747.0, 0.0,      0.0       ],
        [206059.0,       2.678235, 6283.07585],
        [4303.0,         2.6351,   12566.1517],
        [425.0,          1.59,     3.523     ],
        [119.0,          5.796,    26.298    ],
        [109.0,          2.966,    1577.344  ],
        [93.0,           2.59,     18849.23  ],
        [72.0,           1.14,     529.69    ],
        [68.0,           1.87,     398.15    ],
        [67.0,           4.41,     5507.55   ],
        [59.0,           2.89,     5223.69   ],
        [56.0,           2.17,     155.42    ],
        [45.0,           0.4,      796.3     ],
        [36.0,           0.47,     775.52    ],
        [29.0,           2.65,     7.11      ],
        [21.0,           5.34,     0.98      ],
        [19.0,           1.85,     5486.78   ],
        [19.0,           4.97,     213.3     ],
        [17.0,           2.99,     6275.96   ],
        [16.0,           0.03,     2544.31   ],
        [16.0,           1.43,     2146.17   ],
        [15.0,           1.21,     10977.08  ],
        [12.0,           2.83,     1748.02   ],
        [12.0,           3.26,     5088.63   ],
        [12.0,           5.27,     1194.45   ],
        [12.0,           2.08,     4694.0    ],
        [11.0,           0.77,     553.57    ],
        [10.0,           1.3,      3286.6    ],
        [10.0,           4.24,     1349.87   ],
        [9.0,            2.7,      242.73    ],
        [9.0,            5.64,     951.72    ],
        [8.0,            5.3,      2352.87   ],
        [6.0,            2.65,     9437.76   ],
        [6.0,            4.67,     4690.48   ],
    ])

    L2_terms = np.array([
        [52919.0, 0.0,    0.0      ],
        [8720.0,  1.0721, 6283.0758],
        [309.0,   0.867,  12566.152],
        [27.0,    0.05,   3.52     ],
        [16.0,    5.19,   26.3     ],
        [16.0,    3.68,   155.42   ],
        [10.0,    0.76,   18849.23 ],
        [9.0,     2.06,   77713.77 ],
        [7.0,     0.83,   775.52   ],
        [5.0,     4.66,   1577.34  ],
        [4.0,     1.03,   7.11     ],
        [4.0,     3.44,   5573.14  ],
        [3.0,     5.14,   796.3    ],
        [3.0,     6.05,   5507.55  ],
        [3.0,     1.19,   242.73   ],
        [3.0,     6.12,   529.69   ],
        [3.0,     0.31,   398.15   ],
        [3.0,     2.28,   553.57   ],
        [2.0,     4.38,   5223.69  ],
        [2.0,     3.75,   0.98     ],
    ])

    L3_terms = np.array([
        [289.0, 5.844, 6283.076 ],
        [35.0,  0.0,   0.0      ],
        [17.0,  5.49,  12566.15 ],
        [3.0,   5.2,   155.42   ],
        [1.0,   4.72,  3.52     ],
        [1.0,   5.3,   18849.23 ],
        [1.0,   5.97,  242.73   ],
    ])

    L4_terms = np.array([
        [114.0, 3.142, 0.0      ],
        [8.0,   4.13,  6283.08  ],
        [1.0,   3.84,  12566.15 ],
    ])

    L5_terms = np.array([
        [1.0, 3.14, 0.0],
    ])

    def series_sum(terms, jme):
        A = terms[:, 0]; B = terms[:, 1]; C = terms[:, 2]
        return np.sum(A * np.cos(B + C * jme))

    L0 = series_sum(L0_terms, JME)
    L1 = series_sum(L1_terms, JME)
    L2 = series_sum(L2_terms, JME)
    L3 = series_sum(L3_terms, JME)
    L4 = series_sum(L4_terms, JME)
    L5 = series_sum(L5_terms, JME)   # single-row — same formula applies

    longitude = (L0 + L1*JME + L2*JME**2 + L3*JME**3 + L4*JME**4 + L5*JME**5) / 1e8
    longitude = longitude * 180.0 / np.pi                          # radians → degrees
    longitude = _set_to_range(longitude, 0.0, 360.0)

    # ------------------------------------------------------------------
    # LATITUDE terms (B0, B1)
    # ------------------------------------------------------------------
    B0_terms = np.array([
        [280.0, 3.199, 84334.662],
        [102.0, 5.422, 5507.553 ],
        [80.0,  3.88,  5223.69  ],
        [44.0,  3.7,   2352.87  ],
        [32.0,  4.0,   1577.34  ],
    ])

    B1_terms = np.array([
        [9.0, 3.9,  5507.55],
        [6.0, 1.73, 5223.69],
    ])

    B0 = series_sum(B0_terms, JME)
    B1 = series_sum(B1_terms, JME)

    latitude = (B0 + B1*JME) / 1e8
    latitude  = latitude * 180.0 / np.pi                           # radians → degrees
    latitude  = _set_to_range(latitude, 0.0, 360.0)

    # ------------------------------------------------------------------
    # RADIUS VECTOR terms (R0 … R4)
    # ------------------------------------------------------------------
    R0_terms = np.array([
        [100013989.0, 0.0,      0.0       ],
        [1670700.0,   3.0984635,6283.07585],
        [13956.0,     3.05525,  12566.1517],
        [3084.0,      5.1985,   77713.7715],
        [1628.0,      1.1739,   5753.3849 ],
        [1576.0,      2.8469,   7860.4194 ],
        [925.0,       5.453,    11506.77  ],
        [542.0,       4.564,    3930.21   ],
        [472.0,       3.661,    5884.927  ],
        [346.0,       0.964,    5507.553  ],
        [329.0,       5.9,      5223.694  ],
        [307.0,       0.299,    5573.143  ],
        [243.0,       4.273,    11790.629 ],
        [212.0,       5.847,    1577.344  ],
        [186.0,       5.022,    10977.079 ],
        [175.0,       3.012,    18849.228 ],
        [110.0,       5.055,    5486.778  ],
        [98.0,        0.89,     6069.78   ],
        [86.0,        5.69,     15720.84  ],
        [86.0,        1.27,     161000.69 ],
        [85.0,        0.27,     17260.15  ],
        [63.0,        0.92,     529.69    ],
        [57.0,        2.01,     83996.85  ],
        [56.0,        5.24,     71430.7   ],
        [49.0,        3.25,     2544.31   ],
        [47.0,        2.58,     775.52    ],
        [45.0,        5.54,     9437.76   ],
        [43.0,        6.01,     6275.96   ],
        [39.0,        5.36,     4694.0    ],
        [38.0,        2.39,     8827.39   ],
        [37.0,        0.83,     19651.05  ],
        [37.0,        4.9,      12139.55  ],
        [36.0,        1.67,     12036.46  ],
        [35.0,        1.84,     2942.46   ],
        [33.0,        0.24,     7084.9    ],
        [32.0,        0.18,     5088.63   ],
        [32.0,        1.78,     398.15    ],
        [28.0,        1.21,     6286.6    ],
        [28.0,        1.9,      6279.55   ],
        [26.0,        4.59,     10447.39  ],
    ])

    R1_terms = np.array([
        [103019.0, 1.10749, 6283.07585],
        [1721.0,   1.0644,  12566.1517],
        [702.0,    3.142,   0.0       ],
        [32.0,     1.02,    18849.23  ],
        [31.0,     2.84,    5507.55   ],
        [25.0,     1.32,    5223.69   ],
        [18.0,     1.42,    1577.34   ],
        [10.0,     5.91,    10977.08  ],
        [9.0,      1.42,    6275.96   ],
        [9.0,      0.27,    5486.78   ],
    ])

    R2_terms = np.array([
        [4359.0, 5.7846, 6283.0758],
        [124.0,  5.579,  12566.152],
        [12.0,   3.14,   0.0      ],
        [9.0,    3.63,   77713.77 ],
        [6.0,    1.87,   5573.14  ],
        [3.0,    5.47,   18849.0  ],
    ])

    R3_terms = np.array([
        [145.0, 4.273, 6283.076 ],
        [7.0,   3.92,  12566.15 ],
    ])

    R4_terms = np.array([
        [4.0, 2.56, 6283.08],
    ])

    R0 = series_sum(R0_terms, JME)
    R1 = series_sum(R1_terms, JME)
    R2 = series_sum(R2_terms, JME)
    R3 = series_sum(R3_terms, JME)
    R4 = series_sum(R4_terms, JME)

    radius = (R0 + R1*JME + R2*JME**2 + R3*JME**3 + R4*JME**4) / 1e8  # AU

    return {'longitude': longitude, 'latitude': latitude, 'radius': radius}


def _sun_geocentric_position_calculation(earth_helio):
    """
    Compute sun geocentric longitude and latitude from Earth heliocentric position.
    """
    lon = earth_helio['longitude'] + 180.0
    lon = _set_to_range(lon, 0.0, 360.0)

    lat = -earth_helio['latitude']
    lat = _set_to_range(lat, 0.0, 360.0)

    return {'longitude': lon, 'latitude': lat}


def _nutation_calculation(julian):
    """
    Compute nutation in longitude and obliquity (degrees) using
    tabulated Y_terms and nutation_terms (63-row IAU 1980 series).
    """
    JCE = julian['ephemeris_century']

    # Five fundamental arguments (degrees), evaluated as cubic polynomials
    # 1. Mean elongation of the moon from the sun
    p = [1.0/189474.0, -0.0019142, 445267.11148, 297.85036]
    X0 = p[0]*JCE**3 + p[1]*JCE**2 + p[2]*JCE + p[3]

    # 2. Mean anomaly of the sun (earth)
    p = [-1.0/300000.0, -0.0001603, 35999.05034, 357.52772]
    X1 = p[0]*JCE**3 + p[1]*JCE**2 + p[2]*JCE + p[3]

    # 3. Mean anomaly of the moon
    p = [1.0/56250.0, 0.0086972, 477198.867398, 134.96298]
    X2 = p[0]*JCE**3 + p[1]*JCE**2 + p[2]*JCE + p[3]

    # 4. Moon argument of latitude
    p = [1.0/327270.0, -0.0036825, 483202.017538, 93.27191]
    X3 = p[0]*JCE**3 + p[1]*JCE**2 + p[2]*JCE + p[3]

    # 5. Longitude of ascending node of moon's mean orbit on the ecliptic
    p = [1.0/450000.0, 0.0020708, -1934.136261, 125.04452]
    X4 = p[0]*JCE**3 + p[1]*JCE**2 + p[2]*JCE + p[3]

    Xi = np.array([X0, X1, X2, X3, X4])

    # Tabulated Y_terms: multipliers for each of the five Xi arguments (63 rows)
    Y_terms = np.array([
        [ 0,  0,  0,  0,  1],
        [-2,  0,  0,  2,  2],
        [ 0,  0,  0,  2,  2],
        [ 0,  0,  0,  0,  2],
        [ 0,  1,  0,  0,  0],
        [ 0,  0,  1,  0,  0],
        [-2,  1,  0,  2,  2],
        [ 0,  0,  0,  2,  1],
        [ 0,  0,  1,  2,  2],
        [-2, -1,  0,  2,  2],
        [-2,  0,  1,  0,  0],
        [-2,  0,  0,  2,  1],
        [ 0,  0, -1,  2,  2],
        [ 2,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  1],
        [ 2,  0, -1,  2,  2],
        [ 0,  0, -1,  0,  1],
        [ 0,  0,  1,  2,  1],
        [-2,  0,  2,  0,  0],
        [ 0,  0, -2,  2,  1],
        [ 2,  0,  0,  2,  2],
        [ 0,  0,  2,  2,  2],
        [ 0,  0,  2,  0,  0],
        [-2,  0,  1,  2,  2],
        [ 0,  0,  0,  2,  0],
        [-2,  0,  0,  2,  0],
        [ 0,  0, -1,  2,  1],
        [ 0,  2,  0,  0,  0],
        [ 2,  0, -1,  0,  1],
        [-2,  2,  0,  2,  2],
        [ 0,  1,  0,  0,  1],
        [-2,  0,  1,  0,  1],
        [ 0, -1,  0,  0,  1],
        [ 0,  0,  2, -2,  0],
        [ 2,  0, -1,  2,  1],
        [ 2,  0,  1,  2,  2],
        [ 0,  1,  0,  2,  2],
        [-2,  1,  1,  0,  0],
        [ 0, -1,  0,  2,  2],
        [ 2,  0,  0,  2,  1],
        [ 2,  0,  1,  0,  0],
        [-2,  0,  2,  2,  2],
        [-2,  0,  1,  2,  1],
        [ 2,  0, -2,  0,  1],
        [ 2,  0,  0,  0,  1],
        [ 0, -1,  1,  0,  0],
        [-2, -1,  0,  2,  1],
        [-2,  0,  0,  0,  1],
        [ 0,  0,  2,  2,  1],
        [-2,  0,  2,  0,  1],
        [-2,  1,  0,  2,  1],
        [ 0,  0,  1, -2,  0],
        [-1,  0,  1,  0,  0],
        [-2,  1,  0,  0,  0],
        [ 1,  0,  0,  0,  0],
        [ 0,  0,  1,  2,  0],
        [ 0,  0, -2,  2,  2],
        [-1, -1,  1,  0,  0],
        [ 0,  1,  1,  0,  0],
        [ 0, -1,  1,  2,  2],
        [ 2, -1, -1,  2,  2],
        [ 0,  0,  3,  2,  2],
        [ 2, -1,  0,  2,  2],
    ], dtype=float)

    # Tabulated nutation_terms: [a, b, c, d] for each row (63 rows)
    # delta_longitude = (a + b*JCE) * sin(arg)  [units: 0.0001 arcsec]
    # delta_obliquity = (c + d*JCE) * cos(arg)  [units: 0.0001 arcsec]
    nutation_terms = np.array([
        [-171996.0, -174.2,  92025.0,  8.9],
        [ -13187.0,   -1.6,   5736.0, -3.1],
        [  -2274.0,   -0.2,    977.0, -0.5],
        [   2062.0,    0.2,   -895.0,  0.5],
        [   1426.0,   -3.4,     54.0, -0.1],
        [    712.0,    0.1,     -7.0,  0.0],
        [   -517.0,    1.2,    224.0, -0.6],
        [   -386.0,   -0.4,    200.0,  0.0],
        [   -301.0,    0.0,    129.0, -0.1],
        [    217.0,   -0.5,    -95.0,  0.3],
        [   -158.0,    0.0,      0.0,  0.0],
        [    129.0,    0.1,    -70.0,  0.0],
        [    123.0,    0.0,    -53.0,  0.0],
        [     63.0,    0.0,      0.0,  0.0],
        [     63.0,    0.1,    -33.0,  0.0],
        [    -59.0,    0.0,     26.0,  0.0],
        [    -58.0,   -0.1,     32.0,  0.0],
        [    -51.0,    0.0,     27.0,  0.0],
        [     48.0,    0.0,      0.0,  0.0],
        [     46.0,    0.0,    -24.0,  0.0],
        [    -38.0,    0.0,     16.0,  0.0],
        [    -31.0,    0.0,     13.0,  0.0],
        [     29.0,    0.0,      0.0,  0.0],
        [     29.0,    0.0,    -12.0,  0.0],
        [     26.0,    0.0,      0.0,  0.0],
        [    -22.0,    0.0,      0.0,  0.0],
        [     21.0,    0.0,    -10.0,  0.0],
        [     17.0,   -0.1,      0.0,  0.0],
        [     16.0,    0.0,     -8.0,  0.0],
        [    -16.0,    0.1,      7.0,  0.0],
        [    -15.0,    0.0,      9.0,  0.0],
        [    -13.0,    0.0,      7.0,  0.0],
        [    -12.0,    0.0,      6.0,  0.0],
        [     11.0,    0.0,      0.0,  0.0],
        [    -10.0,    0.0,      5.0,  0.0],
        [     -8.0,    0.0,      3.0,  0.0],
        [      7.0,    0.0,     -3.0,  0.0],
        [     -7.0,    0.0,      0.0,  0.0],
        [     -7.0,    0.0,      3.0,  0.0],
        [     -7.0,    0.0,      3.0,  0.0],
        [      6.0,    0.0,      0.0,  0.0],
        [      6.0,    0.0,     -3.0,  0.0],
        [      6.0,    0.0,     -3.0,  0.0],
        [     -6.0,    0.0,      3.0,  0.0],
        [     -6.0,    0.0,      3.0,  0.0],
        [      5.0,    0.0,      0.0,  0.0],
        [     -5.0,    0.0,      3.0,  0.0],
        [     -5.0,    0.0,      3.0,  0.0],
        [     -5.0,    0.0,      3.0,  0.0],
        [      4.0,    0.0,      0.0,  0.0],
        [      4.0,    0.0,      0.0,  0.0],
        [      4.0,    0.0,      0.0,  0.0],
        [     -4.0,    0.0,      0.0,  0.0],
        [     -4.0,    0.0,      0.0,  0.0],
        [     -4.0,    0.0,      0.0,  0.0],
        [      3.0,    0.0,      0.0,  0.0],
        [     -3.0,    0.0,      0.0,  0.0],
        [     -3.0,    0.0,      0.0,  0.0],
        [     -3.0,    0.0,      0.0,  0.0],
        [     -3.0,    0.0,      0.0,  0.0],
        [     -3.0,    0.0,      0.0,  0.0],
        [     -3.0,    0.0,      0.0,  0.0],
        [     -3.0,    0.0,      0.0,  0.0],
    ])

    # Tabulated argument for each row: dot product of Y_terms row with Xi vector
    tabulated_argument = (Y_terms @ Xi) * (np.pi / 180.0)

    delta_longitude = (nutation_terms[:, 0] + nutation_terms[:, 1] * JCE) * np.sin(tabulated_argument)
    delta_obliquity = (nutation_terms[:, 2] + nutation_terms[:, 3] * JCE) * np.cos(tabulated_argument)

    return {
        'longitude': np.sum(delta_longitude) / 36000000.0,
        'obliquity': np.sum(delta_obliquity) / 36000000.0
    }


def _true_obliquity_calculation(julian, nutation):
    """
    Compute true obliquity of the ecliptic (degrees).
    Uses 11-coefficient polynomial for mean obliquity (arcseconds),
    then adds nutation in obliquity.
    """
    # Polynomial coefficients (highest degree first), from Laskar (1986)
    p = [2.45, 5.79, 27.87, 7.12, -39.05, -249.67, -51.38, 1999.25, -1.55, -4680.93, 84381.448]

    U = julian['ephemeris_millenium'] / 10.0

    # Evaluate polynomial explicitly (matching MATLAB code exactly)
    mean_obliquity = (p[0]*U**10 + p[1]*U**9 + p[2]*U**8 + p[3]*U**7 +
                      p[4]*U**6  + p[5]*U**5 + p[6]*U**4 + p[7]*U**3 +
                      p[8]*U**2  + p[9]*U    + p[10])

    true_obliquity = (mean_obliquity / 3600.0) + nutation['obliquity']
    return true_obliquity


def _aberration_correction_calculation(earth_helio):
    """
    Compute aberration correction (degrees) as a function of Earth-Sun distance.
    """
    return -20.4898 / (3600.0 * earth_helio['radius'])


def _apparent_sun_longitude_calculation(sun_geo, nutation, aberration_correction):
    """
    Compute apparent sun longitude (degrees).
    """
    return sun_geo['longitude'] + nutation['longitude'] + aberration_correction


def _apparent_stime_at_greenwich_calculation(julian, nutation, true_obliquity):
    """
    Compute apparent sidereal time at Greenwich (degrees).
    """
    JD = julian['day']
    JC = julian['century']

    # Mean sidereal time (degrees)
    mean_stime = (280.46061837
                  + 360.98564736629 * (JD - 2451545.0)
                  + 0.000387933 * JC**2
                  - JC**3 / 38710000.0)

    mean_stime = _set_to_range(mean_stime, 0.0, 360.0)

    apparent_stime = mean_stime + nutation['longitude'] * np.cos(np.deg2rad(true_obliquity))
    return apparent_stime


def _sun_right_ascension_calculation(apparent_sun_longitude, true_obliquity, sun_geo):
    """
    Compute sun right ascension (degrees), range [0, 360].
    """
    asl_rad = np.deg2rad(apparent_sun_longitude)
    obl_rad = np.deg2rad(true_obliquity)
    lat_rad = np.deg2rad(sun_geo['latitude'])

    numerator   = np.sin(asl_rad) * np.cos(obl_rad) - np.tan(lat_rad) * np.sin(obl_rad)
    denominator = np.cos(asl_rad)

    ra = np.rad2deg(np.arctan2(numerator, denominator))
    return _set_to_range(ra, 0.0, 360.0)


def _sun_geocentric_declination_calculation(apparent_sun_longitude, true_obliquity, sun_geo):
    """
    Compute sun geocentric declination (degrees).
    Positive: sun north of celestial equator.
    """
    lat_rad = np.deg2rad(sun_geo['latitude'])
    obl_rad = np.deg2rad(true_obliquity)
    asl_rad = np.deg2rad(apparent_sun_longitude)

    argument = (np.sin(lat_rad) * np.cos(obl_rad) +
                np.cos(lat_rad) * np.sin(obl_rad) * np.sin(asl_rad))

    return np.rad2deg(np.arcsin(argument))


def _observer_local_hour_calculation(apparent_stime, location, sun_ra):
    """
    Compute observer local hour angle (degrees, westward from south), range [0, 360].
    """
    hl = apparent_stime + location['longitude'] - sun_ra
    return _set_to_range(hl, 0.0, 360.0)


def _topocentric_sun_position_calculate(earth_helio, location, observer_local_hour,
                                         sun_ra, sun_geocentric_declination):
    """
    Compute topocentric sun RA, declination, and RA parallax (degrees),
    accounting for observer altitude and latitude.
    """
    # Equatorial horizontal parallax of the sun (degrees)
    eq_horizontal_parallax = 8.794 / (3600.0 * earth_helio['radius'])

    lat_rad = np.deg2rad(location['latitude'])
    ehp_rad = np.deg2rad(eq_horizontal_parallax)
    hl_rad  = np.deg2rad(observer_local_hour)
    dec_rad = np.deg2rad(sun_geocentric_declination)

    # Term u (radians)
    u = np.arctan(0.99664719 * np.tan(lat_rad))

    # Terms x and y
    x = np.cos(u) + (location['altitude'] / 6378140.0) * np.cos(lat_rad)
    y = (0.99664719 * np.sin(u)) + (location['altitude'] / 6378140.0) * np.sin(lat_rad)

    # Parallax in sun right ascension (radians)
    nominator   = -x * np.sin(ehp_rad) * np.sin(hl_rad)
    denominator = np.cos(dec_rad) - x * np.sin(ehp_rad) * np.cos(hl_rad)
    sun_ra_parallax = np.arctan2(nominator, denominator)  # radians

    ra_parallax_deg = np.rad2deg(sun_ra_parallax)

    # Topocentric right ascension (degrees)
    topo_ra = sun_ra + ra_parallax_deg

    # Topocentric declination (degrees)
    nom = (np.sin(dec_rad) - y * np.sin(ehp_rad)) * np.cos(sun_ra_parallax)
    den = np.cos(dec_rad) - x * np.sin(ehp_rad) * np.cos(hl_rad)
    topo_dec = np.rad2deg(np.arctan2(nom, den))

    return {
        'rigth_ascension_parallax': ra_parallax_deg,
        'rigth_ascension':          topo_ra,
        'declination':              topo_dec
    }


def _topocentric_local_hour_calculate(observer_local_hour, topo_sun):
    """
    Compute topocentric local hour angle (degrees).
    """
    return observer_local_hour - topo_sun['rigth_ascension_parallax']


def _sun_topocentric_zenith_angle_calculate(location, topo_sun, topo_local_hour):
    """
    Compute sun zenith and azimuth angles (degrees), including atmospheric
    refraction correction (T=283 K, P=1010 mbar, as in original MATLAB code).
    """
    lat_rad = np.deg2rad(location['latitude'])
    dec_rad = np.deg2rad(topo_sun['declination'])
    tlh_rad = np.deg2rad(topo_local_hour)

    # True topocentric elevation (without refraction)
    argument      = (np.sin(lat_rad) * np.sin(dec_rad) +
                     np.cos(lat_rad) * np.cos(dec_rad) * np.cos(tlh_rad))
    true_elevation = np.rad2deg(np.arcsin(argument))

    # Atmospheric refraction correction (degrees)
    refraction_arg = true_elevation + (10.3 / (true_elevation + 5.11))
    refraction_corr = 1.02 / (60.0 * np.tan(np.deg2rad(refraction_arg)))

    # Apply refraction only when sun is above -5 degrees elevation
    if true_elevation > -5.0:
        apparent_elevation = true_elevation + refraction_corr
    else:
        apparent_elevation = true_elevation

    zenith = 90.0 - apparent_elevation

    # Topocentric azimuth: converted from astronomer (westward from south)
    # to navigation convention (eastward from north) by adding 180 degrees
    nom = np.sin(tlh_rad)
    den = (np.cos(tlh_rad) * np.sin(lat_rad) -
           np.tan(dec_rad) * np.cos(lat_rad))
    azimuth = np.rad2deg(np.arctan2(nom, den)) + 180.0
    azimuth = _set_to_range(azimuth, 0.0, 360.0)

    return {'zenith': zenith, 'azimuth': azimuth}
