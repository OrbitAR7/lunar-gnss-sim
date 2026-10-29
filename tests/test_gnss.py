import numpy as np
import pytest

from lunarnav.constants import R_MEAN
from lunarnav.frames import enu_basis, latlon_to_unit
from lunarnav.gnss import dop_series, fibonacci_sphere, single_point_solution


def _ideal_geometry(el_deg, range_km=6000.0, lat=30.0, lon=40.0):
    """User plus 1 zenith satellite and 3 at elevation el, equally spaced in azimuth."""
    la, lo = np.radians(lat), np.radians(lon)
    user = latlon_to_unit(np.array(la), np.array(lo))
    enu = enu_basis(la, lo)
    e = np.radians(el_deg)
    dirs = [np.array([0.0, 0.0, 1.0])] + [
        np.array([np.cos(e) * np.sin(az), np.cos(e) * np.cos(az), np.sin(e)])
        for az in np.radians([0.0, 120.0, 240.0])
    ]
    sats = np.array([user * R_MEAN + range_km * (d @ enu) for d in dirs])
    return user, sats


def test_dop_matches_closed_form_for_symmetric_constellation():
    el = 25.0
    user, sats = _ideal_geometry(el)
    n_vis, dop = dop_series(user[None, :], sats[:, None, :], mask_deg=5.0)
    s = np.sin(np.radians(el))
    uu, ut, tt = 1 + 3 * s**2, -(1 + 3 * s), 4.0
    det = uu * tt - ut**2
    assert n_vis[0, 0] == 4
    assert dop["hdop"][0, 0] == pytest.approx(np.sqrt(4 / 3) / np.cos(np.radians(el)), rel=1e-9)
    assert dop["vdop"][0, 0] == pytest.approx(np.sqrt(tt / det), rel=1e-9)
    assert dop["tdop"][0, 0] == pytest.approx(np.sqrt(uu / det), rel=1e-9)
    assert dop["pdop"][0, 0] == pytest.approx(np.hypot(dop["hdop"][0, 0], dop["vdop"][0, 0]))


def test_elevation_mask_removes_low_satellites():
    user, sats = _ideal_geometry(8.0)
    nv_lo, _ = dop_series(user[None, :], sats[:, None, :], mask_deg=5.0)
    nv_hi, dop_hi = dop_series(user[None, :], sats[:, None, :], mask_deg=10.0)
    assert nv_lo[0, 0] == 4 and nv_hi[0, 0] == 1
    assert np.isnan(dop_hi["pdop"][0, 0])


def test_dop_is_nan_below_four_satellites():
    user, sats = _ideal_geometry(30.0)
    nv, dop = dop_series(user[None, :], sats[:3, None, :])
    assert nv[0, 0] == 3 and np.isnan(dop["gdop"][0, 0])


def test_spp_recovers_truth_without_noise_and_errors_scale_with_pdop():
    rng = np.random.default_rng(7)
    user_unit = latlon_to_unit(np.radians(-86.0), np.radians(25.0))
    user = user_unit * R_MEAN
    # six satellites at random positions above a 10 deg mask, 5000-15000 km away
    sats = []
    enu = enu_basis(np.radians(-86.0), np.radians(25.0))
    while len(sats) < 6:
        az, el = rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(10, 85))
        d = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)]) @ enu
        sats.append(user + rng.uniform(5000, 15000) * d)
    sats = np.array(sats)
    bias = 0.003  # km (3 m of range from the receiver clock)
    truth = np.concatenate([user, [bias]])
    pr = np.linalg.norm(sats - user, axis=1) + bias

    sol = single_point_solution(sats, pr, x0=np.concatenate([user * 0.9, [0.0]]))
    np.testing.assert_allclose(sol, truth, atol=1e-9)

    _, dop = dop_series(user_unit[None, :], sats[:, None, :], mask_deg=10.0)
    sigma = 0.001  # 1 m UERE
    errs = np.array([
        np.linalg.norm(single_point_solution(sats, pr + rng.normal(0, sigma, 6),
                                             x0=np.concatenate([user, [0.0]]))[:3] - user)
        for _ in range(4000)
    ])
    assert np.sqrt(np.mean(errs**2)) == pytest.approx(dop["pdop"][0, 0] * sigma, rel=0.05)


def test_fibonacci_sphere_is_uniform():
    pts = fibonacci_sphere(2000)
    np.testing.assert_allclose(np.linalg.norm(pts, axis=1), 1.0)
    assert np.abs(pts.mean(axis=0)).max() < 1e-3
    assert abs((pts[:, 2] > 0.5).mean() - 0.25) < 0.01      # cap above z=0.5 holds 25 % of area
