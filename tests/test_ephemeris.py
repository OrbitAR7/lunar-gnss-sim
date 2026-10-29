import numpy as np
import pytest

from lunarnav.constants import OBLIQUITY_J2000
from lunarnav.ephemeris import Ephemeris, moon_geocentric, sun_geocentric


def _ecliptic_lonlat_deg(r_icrf):
    c, s = np.cos(OBLIQUITY_J2000), np.sin(OBLIQUITY_J2000)
    x, y, z = r_icrf[0], c * r_icrf[1] + s * r_icrf[2], -s * r_icrf[1] + c * r_icrf[2]
    return np.degrees(np.arctan2(y, x)) % 360, np.degrees(np.arctan2(z, np.hypot(x, y)))


def _precession_deg(jd):
    return 1.3969713 * (jd - 2451545.0) / 36525.0   # general precession in longitude


def test_moon_vs_meeus_example_47a():
    # Meeus, Astronomical Algorithms ch. 47: 1992 April 12, 0h TD
    #   geocentric lon 133.162655 deg, lat -3.229126 deg, distance 368409.7 km
    jd = 2448724.5
    lon, lat = _ecliptic_lonlat_deg(moon_geocentric(jd))
    lon_of_date = lon + _precession_deg(jd)
    assert lon_of_date == pytest.approx(133.162655, abs=0.15)
    assert lat == pytest.approx(-3.229126, abs=0.15)
    assert np.linalg.norm(moon_geocentric(jd)) == pytest.approx(368409.7, rel=0.005)


def test_sun_vs_meeus_example_25a():
    # Meeus ch. 25: 1992 October 13, 0h TD: apparent lon 199.90895 deg, R = 0.99760775 AU
    jd = 2448908.5
    r = sun_geocentric(jd)
    lon, lat = _ecliptic_lonlat_deg(r)
    assert lon + _precession_deg(jd) == pytest.approx(199.90895, abs=0.15)
    assert abs(lat) < 0.01
    assert np.linalg.norm(r) / 149597870.7 == pytest.approx(0.99760775, rel=0.002)


def test_moon_distance_stays_within_perigee_apogee():
    jds = 2461041.5 + np.arange(0, 400, 0.25)
    d = np.array([np.linalg.norm(moon_geocentric(j)) for j in jds])
    assert 355000 < d.min() < 365000
    assert 400000 < d.max() < 408000


def test_spline_cache_matches_direct_series():
    jd0 = 2461041.5
    eph = Ephemeris(jd0, 5.0)
    for days in (0.3, 2.71, 4.9):
        direct = -moon_geocentric(jd0 + days)
        np.testing.assert_allclose(eph.earth_wrt_moon(days * 86400.0), direct, atol=1.0)
