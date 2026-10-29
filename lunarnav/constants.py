"""Physical constants. Units are km, s, rad unless stated otherwise.

Sources
-------
GM_MOON, R_REF, J2/C22/degree-3 harmonics: GRAIL-era solution as tabulated on
    https://en.wikipedia.org/wiki/Gravitation_of_the_Moon (secondary summary of
    Konopliv+2014, Lemoine+2013, Williams+2014). Coefficients are unnormalised and
    expressed in the *principal-axis* frame, which is why C21 = S21 = S22 = 0.
    The degree-3 values come from that same secondary table; swap in a full
    GRGM1200A/LP165P table if you need more than a demonstration.
GM_EARTH, GM_SUN: JPL DE440 (Park et al. 2021), rounded.
R_MEAN: IAU mean lunar radius (Archinal et al. 2018).
PA_TO_ME_ARCSEC: NAIF frame kernel moon_de440_*.tf, frame MOON_ME_DE440_ME421.
"""
import math

DEG = math.pi / 180.0
ARCSEC = DEG / 3600.0

# --- lunar gravity ---------------------------------------------------------
GM_MOON = 4902.8001          # km^3 / s^2
R_REF = 1738.0               # km, reference radius of the gravity field
R_MEAN = 1737.4              # km, IAU mean radius (used for the spherical user surface)

# Unnormalised C_nm, S_nm in the principal-axis frame (C20 = -J2).
J2 = 203.3e-6
C22 = 22.4e-6
J3 = 8.46e-6
C31, S31 = 28.48e-6, 5.89e-6
C32, S32 = 4.84e-6, 1.67e-6
C33, S33 = 1.71e-6, -0.25e-6

# --- third bodies ----------------------------------------------------------
GM_EARTH = 398600.4355       # km^3 / s^2
GM_SUN = 1.32712440041e11    # km^3 / s^2

# --- time ------------------------------------------------------------------
JD_J2000 = 2451545.0
SECONDS_PER_DAY = 86400.0
TAI_MINUS_UTC = 37.0         # s, valid from 2017-01-01 until the next leap second
TT_MINUS_TAI = 32.184        # s

# --- lunar frames ----------------------------------------------------------
# NAIF frame kernel: TKFRAME_31009_ANGLES (67.8526, 78.6944, 0.2785) arcsec with axes
# (3, 2, 1). NAIF states these are the *inverse* of the cited PA->ME rotation, i.e. the
# cited rotation is Rx(-0.2785") Ry(-78.6944") Rz(-67.8526") applied to PA vectors.
PA_TO_ME_ARCSEC = (-0.2785, -78.6944, -67.8526)   # (x, y, z) angles, applied as Rx Ry Rz

OBLIQUITY_J2000 = 23.43929111 * DEG
C_LIGHT = 299792.458         # km / s
