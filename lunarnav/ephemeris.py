"""Low-precision analytic Moon and Sun ephemerides (no kernels required).

Moon: truncated Brown/ELP series from Montenbruck & Gill, "Satellite Orbits" (2000), 3.3.2.
Sun : "MiniSun" series from the same section.

Both return geocentric positions in the ICRF/J2000 equatorial axes (the ecliptic series
are referred to the J2000 equinox, then rotated by the J2000 mean obliquity).

Accuracy is a few tenths of a degree in direction and ~0.5 % in lunar distance. That is
adequate for third-body perturbations and for checking lunar libration amplitudes, but
it is NOT adequate for precise orbit determination. The ``Ephemeris`` interface is
deliberately tiny so a SPICE-backed implementation can replace it.
"""
import numpy as np
from scipy.interpolate import CubicSpline

from .constants import ARCSEC, JD_J2000, OBLIQUITY_J2000
from .timeutil import centuries_since_j2000

_ARC = 1296000.0  # arcsec per revolution


def _frac(x):
    return x - np.floor(x)


def _ecl_to_icrf(r_ecl):
    c, s = np.cos(OBLIQUITY_J2000), np.sin(OBLIQUITY_J2000)
    x, y, z = r_ecl
    return np.array([x, c * y - s * z, s * y + c * z])


def moon_geocentric(jd):
    """Geocentric Moon position [km], ICRF axes, from a truncated lunar series."""
    t = float(centuries_since_j2000(jd))
    l0 = _frac(0.606433 + 1336.851344 * t)              # mean longitude [rev]
    l = 2 * np.pi * _frac(0.374897 + 1325.552410 * t)   # Moon mean anomaly
    lp = 2 * np.pi * _frac(0.993133 + 99.997361 * t)    # Sun mean anomaly
    d = 2 * np.pi * _frac(0.827361 + 1236.853086 * t)   # mean elongation
    f = 2 * np.pi * _frac(0.259086 + 1342.227825 * t)   # argument of latitude

    dl = (22640 * np.sin(l) - 4586 * np.sin(l - 2 * d) + 2370 * np.sin(2 * d)
          + 769 * np.sin(2 * l) - 668 * np.sin(lp) - 412 * np.sin(2 * f)
          - 212 * np.sin(2 * l - 2 * d) - 206 * np.sin(l + lp - 2 * d)
          + 192 * np.sin(l + 2 * d) - 165 * np.sin(lp - 2 * d) - 125 * np.sin(d)
          - 110 * np.sin(l + lp) + 148 * np.sin(l - lp) - 55 * np.sin(2 * f - 2 * d))
    s = f + (dl + 412 * np.sin(2 * f) + 541 * np.sin(lp)) * ARCSEC
    h = f - 2 * d
    n = (-526 * np.sin(h) + 44 * np.sin(l + h) - 31 * np.sin(-l + h) - 23 * np.sin(lp + h)
         + 11 * np.sin(-lp + h) - 25 * np.sin(-2 * l + f) + 21 * np.sin(-l + f))

    lon = 2 * np.pi * l0 + dl * ARCSEC
    lat = (18520.0 * np.sin(s) + n) * ARCSEC
    r = (385000.0 - 20905 * np.cos(l) - 3699 * np.cos(2 * d - l) - 2956 * np.cos(2 * d)
         - 570 * np.cos(2 * l) + 246 * np.cos(2 * l - 2 * d) - 205 * np.cos(lp - 2 * d)
         - 171 * np.cos(l + 2 * d) - 152 * np.cos(l + lp - 2 * d))
    r_ecl = r * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return _ecl_to_icrf(r_ecl)


def sun_geocentric(jd):
    """Geocentric Sun position [km], ICRF axes, from the MiniSun series."""
    t = float(centuries_since_j2000(jd))
    m = 2 * np.pi * _frac(0.9931267 + 99.9973583 * t)
    lon = 2 * np.pi * _frac(0.7859444 + m / (2 * np.pi)
                            + (6892.0 * np.sin(m) + 72.0 * np.sin(2 * m)) / _ARC)
    r = 149.619e6 - 2.499e6 * np.cos(m) - 0.021e6 * np.cos(2 * m)
    return _ecl_to_icrf(r * np.array([np.cos(lon), np.sin(lon), 0.0]))


class Ephemeris:
    """Earth and Sun positions relative to the Moon, ICRF axes, cached on a spline grid.

    Evaluating the lunar series inside an ODE right-hand side is wasteful; positions vary
    smoothly, so a cubic spline on a 30 min grid reproduces the series to well below its
    own error.
    """

    def __init__(self, jd_start, duration_days, step_minutes=30.0):
        n = int(np.ceil(duration_days * 1440.0 / step_minutes)) + 3
        jds = jd_start - step_minutes / 1440.0 + np.arange(n) * step_minutes / 1440.0
        self._t0 = jd_start
        secs = (jds - jd_start) * 86400.0
        earth = np.array([-moon_geocentric(j) for j in jds])
        sun = np.array([sun_geocentric(j) - moon_geocentric(j) for j in jds])
        self._earth = CubicSpline(secs, earth)
        self._sun = CubicSpline(secs, sun)

    def earth_wrt_moon(self, t):
        """Earth position relative to the Moon [km] at ``t`` seconds after ``jd_start``."""
        return self._earth(t)

    def sun_wrt_moon(self, t):
        return self._sun(t)
