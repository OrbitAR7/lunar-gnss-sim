import numpy as np
import pytest
from scipy.special import lpmv

from lunarnav import constants as C
from lunarnav.gravity import default_coefficients, harmonic_acceleration, third_body_acceleration

R_TEST = np.array([2200.0, -1500.0, 2900.0])  # km, arbitrary off-axis point


def _potential(r_bf, cnm, snm):
    """Independent potential from associated Legendre functions (no recursion shared)."""
    x, y, z = r_bf
    r = np.sqrt(x * x + y * y + z * z)
    sphi, lam = z / r, np.arctan2(y, x)
    u = 0.0
    for n in range(cnm.shape[0]):
        for m in range(n + 1):
            pnm = (-1) ** m * lpmv(m, n, sphi)  # drop the Condon-Shortley phase
            u += (C.R_REF / r) ** n * pnm * (cnm[n, m] * np.cos(m * lam) + snm[n, m] * np.sin(m * lam))
    return C.GM_MOON / r * u


def test_point_mass_limit():
    cnm, snm = default_coefficients(0)
    a = harmonic_acceleration(R_TEST, cnm, snm)
    r = np.linalg.norm(R_TEST)
    np.testing.assert_allclose(a, -C.GM_MOON * R_TEST / r**3, rtol=1e-13)


def test_j2_closed_form():
    cnm, snm = default_coefficients(2)
    cnm[2, 2] = 0.0
    a = harmonic_acceleration(R_TEST, cnm, snm)
    x, y, z = R_TEST
    r = np.linalg.norm(R_TEST)
    k = 1.5 * C.J2 * (C.R_REF / r) ** 2
    expected = -C.GM_MOON / r**3 * np.array([
        x * (1 - k * (5 * z**2 / r**2 - 1)),
        y * (1 - k * (5 * z**2 / r**2 - 1)),
        z * (1 - k * (5 * z**2 / r**2 - 3)),
    ])
    np.testing.assert_allclose(a, expected, rtol=1e-12)


def test_degree3_matches_finite_difference_of_potential():
    cnm, snm = default_coefficients(3)
    h = 1e-3
    grad = np.array([
        (_potential(R_TEST + h * e, cnm, snm) - _potential(R_TEST - h * e, cnm, snm)) / (2 * h)
        for e in np.eye(3)
    ])
    a = harmonic_acceleration(R_TEST, cnm, snm)
    np.testing.assert_allclose(a, grad, rtol=1e-6, atol=1e-12)


def test_third_body_direct_plus_indirect_on_the_earth_line():
    # On the Moon-Earth line the exact result is GM [1/(D-r)^2 - 1/D^2] ~ 2 GM r / D^3.
    d_earth, r_sc = 384400.0, 6000.0
    a = third_body_acceleration(np.array([r_sc, 0.0, 0.0]), np.array([d_earth, 0.0, 0.0]), C.GM_EARTH)
    exact = C.GM_EARTH * (1 / (d_earth - r_sc) ** 2 - 1 / d_earth**2)
    assert a[0] == pytest.approx(exact, rel=1e-12)
    assert a[0] == pytest.approx(2 * C.GM_EARTH * r_sc / d_earth**3, rel=0.06)
    assert a[1] == 0.0 and a[2] == 0.0
