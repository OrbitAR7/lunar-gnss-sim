from datetime import datetime, timezone

import numpy as np
import pytest

from lunarnav import constants as C
from lunarnav.constellation import clfo, s_elfo
from lunarnav.orbits import (cartesian_to_kepler, frozen_eccentricity, kepler_to_cartesian,
                             period, semi_major_axis_from_period)
from lunarnav.propagate import ForceModel, elements_to_mci, mci_to_elements, propagate
from lunarnav.timeutil import jd_tt_from_utc

JD0 = jd_tt_from_utc(datetime(2027, 1, 1, tzinfo=timezone.utc))
DEG = np.pi / 180


def test_kepler_roundtrip():
    el = (9750.7, 0.672, 55 * DEG, 30 * DEG, 90 * DEG, 1.234)
    r, v = kepler_to_cartesian(*el)
    np.testing.assert_allclose(cartesian_to_kepler(r, v), el, rtol=1e-11, atol=1e-11)


def test_vis_viva_and_period():
    a = semi_major_axis_from_period(86400.0)
    assert period(a) == pytest.approx(86400.0)
    r, v = kepler_to_cartesian(a, 0.6, 1.0, 0.3, 0.5, 2.0)
    assert np.dot(v, v) / 2 - C.GM_MOON / np.linalg.norm(r) == pytest.approx(-C.GM_MOON / (2 * a))


def test_frozen_eccentricity_limits():
    assert frozen_eccentricity(np.pi / 2) == pytest.approx(1.0)
    assert frozen_eccentricity(np.arccos(np.sqrt(0.6)) + 1e-9) < 1e-3     # ~39.23 deg: circular
    assert frozen_eccentricity(55 * DEG) == pytest.approx(0.6721, abs=1e-4)
    with pytest.raises(ValueError):
        frozen_eccentricity(30 * DEG)


def test_two_body_closes_after_one_period_and_conserves_energy():
    a = 6142.0
    el = (a, 0.6, 50 * DEG, 10 * DEG, 20 * DEG, 0.0)
    r0, v0 = elements_to_mci(el, JD0)
    t = np.array([0.0, period(a) / 3, period(a)])
    st = propagate(r0, v0, JD0, t, ForceModel(n_max=0, earth=False, sun=False), rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(st[-1, :3] - r0) < 1e-5            # < 1 cm
    energy = [np.dot(s[3:], s[3:]) / 2 - C.GM_MOON / np.linalg.norm(s[:3]) for s in st]
    assert np.ptp(energy) / abs(energy[0]) < 1e-11


def test_j2_secular_rates_match_analytic_theory():
    a, e, inc = 4500.0, 0.1, 60 * DEG
    el = (a, e, inc, 40 * DEG, 70 * DEG, 0.0)
    r0, v0 = elements_to_mci(el, JD0, "equator")
    t = np.arange(0.0, 6 * 86400.0, 300.0)
    st = propagate(r0, v0, JD0, t, ForceModel(n_max=2, earth=False, sun=False, zonal_only=True),
                   rtol=1e-12, atol=1e-10)
    # Measure in the equator frame frozen at epoch: the IAU pole itself wobbles by ~0.01 deg/day
    # (it carries the forced libration terms), which would contaminate the node rate.
    elt = mci_to_elements(st, np.full(len(t), JD0), "equator")
    days = t / 86400.0
    raan_rate = np.polyfit(days, np.degrees(np.unwrap(elt[:, 3])), 1)[0]
    argp_rate = np.polyfit(days, np.degrees(np.unwrap(elt[:, 4])), 1)[0]

    a_mean = elt[:, 0].mean()
    n = np.sqrt(C.GM_MOON / a_mean**3)
    p = a_mean * (1 - e * e)
    k = 1.5 * n * C.J2 * (C.R_REF / p) ** 2 * 86400.0 / DEG          # deg/day
    assert raan_rate == pytest.approx(-k * np.cos(inc), rel=0.02)
    assert argp_rate == pytest.approx(k * (2 - 2.5 * np.sin(inc) ** 2), rel=0.02)


def test_s_elfo_stays_frozen_for_two_months_without_station_keeping():
    """The design value e = sqrt(1 - 5/3 cos^2 i) holds the orbit near-frozen against the
    Earth when inclination is measured from the Earth-Moon orbit plane."""
    const = s_elfo(n_sats=3, n_planes=3)
    t = np.arange(0.0, 60 * 86400.0, 3600.0)
    r0, v0 = elements_to_mci(const.elements[0], JD0, const.reference)
    st = propagate(r0, v0, JD0, t, ForceModel())
    elt = mci_to_elements(st, JD0 + t / 86400.0, const.reference)
    e, inc, argp = elt[:, 1], np.degrees(elt[:, 2]), np.degrees(np.unwrap(elt[:, 4]))
    assert 0.60 < e.min() and e.max() < 0.73
    assert abs(inc - 55.0).max() < 3.5
    assert abs(argp - 90.0).max() < 10.0


def test_walker_phasing_and_counts():
    c = s_elfo(12, 3, 1)
    assert c.n_sat == 12
    raans = sorted({round(el[3], 9) for el in c.elements})
    np.testing.assert_allclose(np.diff(raans), 2 * np.pi / 3)
    # in-plane spacing 90 deg; plane p is advanced by p * F * 360 / T = p * 30 deg
    m = [el[5] for el in c.elements]
    assert (m[1] - m[0]) % (2 * np.pi) == pytest.approx(np.pi / 2)
    assert (m[4] - m[0]) % (2 * np.pi) == pytest.approx(np.radians(30))
    assert clfo().n_sat == 24
