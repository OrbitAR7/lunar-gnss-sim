import numpy as np
import pytest

from lunarnav import constants as C
from lunarnav.ephemeris import moon_geocentric
from lunarnav.frames import (R_icrf_to_equator, R_icrf_to_me, R_icrf_to_pa, R_pa_to_me,
                             iau_moon_angles, latlon_to_unit, unit_to_latlon, _E_RATE)

JD = 2461041.5  # 2027-01-01 00:00 TT, roughly


def test_rotations_are_proper():
    for r in (R_icrf_to_me(JD), R_icrf_to_pa(JD), R_icrf_to_equator(JD), R_pa_to_me()):
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-14)
        assert np.linalg.det(r) == pytest.approx(1.0)


def test_me_z_axis_is_iau_pole():
    alpha0, delta0, _ = iau_moon_angles(JD)
    pole = np.array([np.cos(delta0) * np.cos(alpha0), np.cos(delta0) * np.sin(alpha0), np.sin(delta0)])
    np.testing.assert_allclose(R_icrf_to_me(JD)[2], pole, atol=1e-14)
    np.testing.assert_allclose(R_icrf_to_equator(JD)[2], pole, atol=1e-14)


def test_prime_meridian_rate_matches_iau():
    # Periodic terms change the instantaneous rate by < 0.02 deg/day around 13.17635815.
    for offset in (0.0, 1000.3, 4321.7):
        jd = JD + offset
        dw = np.degrees(iau_moon_angles(jd + 1.0)[2] - iau_moon_angles(jd - 1.0)[2]) % 360.0
        assert dw / 2.0 == pytest.approx(13.17635815, abs=0.03)


def test_precession_rates_agree_with_iau_per_day_values():
    # IAU 2009 report lists E_i rates per day; NAIF lists them per Julian century.
    per_day = [-0.0529921, -0.1059842, 13.0120009, 13.3407154, 0.9856003, 26.4057084,
               13.0649930, 0.3287146, 1.7484877, -0.1589763, 0.0036096, 0.1643573, 12.9590088]
    np.testing.assert_allclose(_E_RATE / 36525.0, per_day, atol=2e-7)


def test_pa_me_offset_magnitude_matches_naif():
    r = R_pa_to_me()
    angle = np.degrees(np.arccos((np.trace(r) - 1) / 2))
    assert angle == pytest.approx(0.02886, abs=2e-5)          # NAIF: ~0.02886 deg
    assert angle * np.pi / 180 * C.R_MEAN * 1e3 == pytest.approx(875.0, abs=15.0)  # NAIF: ~875 m


def test_pa_and_me_share_the_icrf_chain():
    np.testing.assert_allclose(R_pa_to_me() @ R_icrf_to_pa(JD), R_icrf_to_me(JD), atol=1e-14)


def test_latlon_roundtrip():
    lat, lon = np.radians([-89.3, 33.0]), np.radians([120.0, -170.0])
    lat2, lon2 = unit_to_latlon(latlon_to_unit(lat, lon))
    np.testing.assert_allclose(lat2, lat, atol=1e-14)
    np.testing.assert_allclose(lon2, lon, atol=1e-14)


def test_earth_libration_envelope_and_mean_earth_direction():
    """Joint check of frame + ephemeris: the sub-Earth point librates about (0, 0).

    Textbook values: optical libration reaches about +/-7.9 deg in longitude and
    +/-6.7 deg in latitude, centred on the mean Earth direction (the ME +X axis).
    """
    jds = JD + np.arange(0, 19 * 365.25, 0.5)
    lat = np.empty(len(jds))
    lon = np.empty(len(jds))
    for k, jd in enumerate(jds):
        earth_me = R_icrf_to_me(jd) @ (-moon_geocentric(jd))
        la, lo = unit_to_latlon(earth_me)
        lat[k], lon[k] = la, lo
    lon, lat = np.degrees(lon), np.degrees(lat)
    assert abs(lon.mean()) < 0.3 and abs(lat.mean()) < 0.3
    assert 7.0 < np.abs(lon).max() < 8.6
    assert 6.0 < np.abs(lat).max() < 7.4
