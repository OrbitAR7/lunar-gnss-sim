"""Physical constants. Units are km, s, rad unless stated otherwise.

Sources
-------
GM_MOON, R_REF, degree 2-3 harmonics: GSFC GRAIL field GRGM1200A (Lemoine et al. 2014,
    doi:10.1002/2014GL060027), copied digit for digit from the header and first rows of
    the PDS file gggrx_1200a_sha.tab (PDS Geosciences Node, GRAIL archive
    grail-l-lgrs-5-rdr-v1/grail_1001/shadr). The file's coefficients are fully
    normalised and expressed in the DE430 *principal-axis* frame, which is why C21, S21
    and S22 are ~1e-9 rather than exactly zero. The code applies them in the DE440 PA
    frame; the small difference between the DE430 and DE440 PA realisations is ignored.
GM_EARTH, GM_SUN: JPL DE440 (Park et al. 2021), rounded.
R_MEAN: IAU mean lunar radius (Archinal et al. 2018).
PA_TO_ME_ARCSEC: NAIF frame kernel moon_de440_*.tf, frame MOON_ME_DE440_ME421.
"""
import math

DEG = math.pi / 180.0
ARCSEC = DEG / 3600.0

# --- lunar gravity ---------------------------------------------------------
GM_MOON = 4902.8001224453    # km^3 / s^2, GRGM1200A header
R_REF = 1738.0               # km, reference radius of the gravity field
R_MEAN = 1737.4              # km, IAU mean radius (used for the spherical user surface)

# GRGM1200A fully normalised (Cbar_nm, Sbar_nm), principal-axis frame, as in the PDS file.
GRGM1200A_NORMALISED = {
    (2, 0): (-9.0884339347424299e-05, 0.0),
    (2, 1): (1.4664123550281899e-11, 1.1732764234889199e-09),
    (2, 2): (3.4673096470696298e-05, 9.0791898343722900e-10),
    (3, 0): (-3.1973308084610398e-06, 0.0),
    (3, 1): (2.6368006624354499e-05, 5.4545456359614202e-06),
    (3, 2): (1.4171528802855700e-05, 4.8780365791727604e-06),
    (3, 3): (1.2275208368114099e-05, -1.7742221828590300e-06),
}


def normalisation(n, m):
    """Factor N_nm with C_nm = N_nm * Cbar_nm for geodesy 4-pi normalised coefficients."""
    return math.sqrt((2 - (m == 0)) * (2 * n + 1) * math.factorial(n - m) / math.factorial(n + m))


J2 = -normalisation(2, 0) * GRGM1200A_NORMALISED[(2, 0)][0]    # 2.0322e-4
J3 = -normalisation(3, 0) * GRGM1200A_NORMALISED[(3, 0)][0]    # 8.459e-6
C22 = normalisation(2, 2) * GRGM1200A_NORMALISED[(2, 2)][0]    # 2.2381e-5

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
