"""Lunar reference frames.

Frames used in this package
---------------------------
ICRF / MCI
    Axes of the ICRF (J2000 equator); origin at the Moon's centre for state vectors
    ("Moon-centred inertial", MCI). Orbit propagation happens here.
ME  (Mean Earth / Polar Axis)
    Body-fixed. +Z along the mean rotation axis, prime meridian through the mean Earth
    direction. This is the frame of lunar cartography (LRO/LOLA, selenographic lat/lon)
    and the one the IAU orientation model describes. Realised here by the IAU 2009
    series (the form NAIF ships as IAU_MOON in pck00011.tpc).
PA  (Principal Axis)
    Body-fixed. Axes are the Moon's principal axes of inertia. Gravity-field
    coefficients are defined here, so C21 = S21 = S22 = 0 and C20, C22 dominate.
    PA differs from ME by a *constant* rotation of ~0.029 deg (~875 m on the surface).

Rotation convention: ``R_a_to_b`` maps components in frame ``a`` to components in
frame ``b`` (v_b = R @ v_a). Elementary rotations ``rot_x/y/z`` are *frame* (passive)
rotations, the same convention as SPICE's ``rotate``/``eul2m``.
"""
import numpy as np

from .constants import ARCSEC, DEG, OBLIQUITY_J2000, PA_TO_ME_ARCSEC
from .timeutil import centuries_since_j2000, days_since_j2000

INCL_LUNAR_ORBIT = 5.145 * DEG   # mean inclination of the lunar orbit to the ecliptic


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


# --- IAU 2009 lunar orientation (Archinal et al. 2011), as tabulated in NAIF pck00011.tpc.
# Nutation/precession angles E1..E13: (value at J2000 [deg], rate [deg / Julian century]).
_E_EPOCH = np.array([125.045, 250.089, 260.008, 176.625, 357.529, 311.589, 134.963,
                     276.617, 34.226, 15.134, 119.743, 239.961, 25.053])
_E_RATE = np.array([-1935.5364525, -3871.072905, 475263.3328725, 487269.629985,
                    35999.0509575, 964468.49931, 477198.869325, 12006.300765,
                    63863.5132425, -5806.6093575, 131.84064, 6003.1503825,
                    473327.79642])

_RA_COEF = np.array([-3.8787, -0.1204, 0.0700, -0.0172, 0.0, 0.0072, 0.0, 0.0,
                     0.0, -0.0052, 0.0, 0.0, 0.0043])
_DEC_COEF = np.array([1.5419, 0.0239, -0.0278, 0.0068, 0.0, -0.0029, 0.0009, 0.0,
                      0.0, 0.0008, 0.0, 0.0, -0.0009])
_PM_COEF = np.array([3.5610, 0.1208, -0.0642, 0.0158, 0.0252, -0.0066, -0.0047,
                     -0.0046, 0.0028, 0.0052, 0.0040, 0.0019, -0.0044])


def iau_moon_angles(jd):
    """(alpha0, delta0, W) in radians: pole right ascension/declination and prime meridian."""
    t = float(centuries_since_j2000(jd))
    d = float(days_since_j2000(jd))
    e = (_E_EPOCH + _E_RATE * t) * DEG
    s, c = np.sin(e), np.cos(e)
    alpha0 = 269.9949 + 0.0031 * t + _RA_COEF @ s
    delta0 = 66.5392 + 0.0130 * t + _DEC_COEF @ c
    w = 38.3213 + 13.17635815 * d - 1.4e-12 * d * d + _PM_COEF @ s
    return alpha0 * DEG, delta0 * DEG, np.mod(w, 360.0) * DEG


def R_icrf_to_equator(jd):
    """ICRF -> inertial frame whose Z is the IAU lunar pole of date, X on the ICRF-equator node.

    Optional reference for orbit elements (the default is ``R_icrf_to_orbit_plane``). The
    IAU pole carries the forced-libration terms, so this frame wobbles by ~0.01 deg/day;
    freeze it at one epoch when measuring secular element rates.
    """
    alpha0, delta0, _ = iau_moon_angles(jd)
    return rot_x(np.pi / 2 - delta0) @ rot_z(np.pi / 2 + alpha0)


def R_icrf_to_orbit_plane(jd):
    """ICRF -> inertial frame whose Z is the normal of the mean Earth-Moon orbit plane.

    The plane is inclined 5.145 deg to the J2000 ecliptic with its ascending node at the IAU
    argument E1. E1 is the lunar mean node referred to the fixed J2000 equinox: its rate
    (-1935.54 deg/cy) is Meeus' node-of-date rate (-1934.14 deg/cy) minus the 1.40 deg/cy
    general precession. Third-body (Earth) perturbation theory measures a lunar orbit's
    inclination from this plane, so frozen-orbit design uses it as the reference.
    Mean plane only: against DE440 the mean normal agrees to 0.02 deg, and the osculating
    plane departs by up to 0.18 deg (the 173-day inclination oscillation).
    """
    node = np.radians(_E_EPOCH[0] + _E_RATE[0] * float(centuries_since_j2000(jd)))
    return rot_x(INCL_LUNAR_ORBIT) @ rot_z(node) @ rot_x(OBLIQUITY_J2000)


def R_icrf_to_me(jd):
    """ICRF -> Moon ME body-fixed (IAU series)."""
    alpha0, delta0, w = iau_moon_angles(jd)
    return rot_z(w) @ rot_x(np.pi / 2 - delta0) @ rot_z(np.pi / 2 + alpha0)


def R_pa_to_me():
    """Constant rotation PA -> ME (NAIF MOON_ME_DE440_ME421 relative to MOON_PA_DE440)."""
    ax, ay, az = (a * ARCSEC for a in PA_TO_ME_ARCSEC)
    return rot_x(ax) @ rot_y(ay) @ rot_z(az)


def R_icrf_to_pa(jd):
    """ICRF -> Moon PA body-fixed, built as ME rotated back by the constant PA/ME offset."""
    return R_pa_to_me().T @ R_icrf_to_me(jd)


# --- body-fixed coordinates -------------------------------------------------
def latlon_to_unit(lat, lon):
    """Selenographic latitude/longitude [rad] -> unit vector in ME (x to mean Earth, z north)."""
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat) * np.ones_like(cl)], axis=-1)


def unit_to_latlon(v):
    v = np.asarray(v, dtype=float)
    r = np.linalg.norm(v, axis=-1)
    return np.arcsin(v[..., 2] / r), np.arctan2(v[..., 1], v[..., 0])


def enu_basis(lat, lon):
    """Rows are east, north, up unit vectors (in ME) for a spherical-Moon local horizon."""
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    east = np.array([-so, co, 0.0])
    north = np.array([-sl * co, -sl * so, cl])
    up = np.array([cl * co, cl * so, sl])
    return np.stack([east, north, up])
