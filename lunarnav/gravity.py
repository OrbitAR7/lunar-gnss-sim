"""Lunar gravity: spherical-harmonic field in the principal-axis frame plus third bodies.

The harmonic acceleration uses the Cunningham recursion for *unnormalised* coefficients
(Montenbruck & Gill, 3.2.4). It is verified in tests/test_gravity.py against a finite
difference of an independent potential built from scipy's associated Legendre functions.
"""
import numpy as np

from . import constants as C


def default_coefficients(n_max=3):
    """(Cnm, Snm) arrays of shape (n_max+1, n_max+1), unnormalised, C00 = 1."""
    cnm = np.zeros((4, 4))
    snm = np.zeros((4, 4))
    cnm[0, 0] = 1.0
    cnm[2, 0] = -C.J2
    cnm[2, 2] = C.C22
    cnm[3, 0] = -C.J3
    cnm[3, 1], snm[3, 1] = C.C31, C.S31
    cnm[3, 2], snm[3, 2] = C.C32, C.S32
    cnm[3, 3], snm[3, 3] = C.C33, C.S33
    return cnm[:n_max + 1, :n_max + 1], snm[:n_max + 1, :n_max + 1]


def harmonic_acceleration(r_bf, cnm, snm, gm=C.GM_MOON, r_ref=C.R_REF):
    """Acceleration [km/s^2] at body-fixed position ``r_bf`` [km], including the point mass."""
    n_max = cnm.shape[0] - 1
    x, y, z = r_bf
    r2 = x * x + y * y + z * z
    rho = r_ref * r_ref / r2
    x0, y0, z0 = r_ref * x / r2, r_ref * y / r2, r_ref * z / r2

    v = np.zeros((n_max + 2, n_max + 2))
    w = np.zeros((n_max + 2, n_max + 2))
    v[0, 0] = r_ref / np.sqrt(r2)
    for m in range(n_max + 2):
        if m > 0:
            v[m, m] = (2 * m - 1) * (x0 * v[m - 1, m - 1] - y0 * w[m - 1, m - 1])
            w[m, m] = (2 * m - 1) * (x0 * w[m - 1, m - 1] + y0 * v[m - 1, m - 1])
        if m <= n_max:
            v[m + 1, m] = (2 * m + 1) * z0 * v[m, m]
            w[m + 1, m] = (2 * m + 1) * z0 * w[m, m]
        for n in range(m + 2, n_max + 2):
            v[n, m] = ((2 * n - 1) * z0 * v[n - 1, m] - (n + m - 1) * rho * v[n - 2, m]) / (n - m)
            w[n, m] = ((2 * n - 1) * z0 * w[n - 1, m] - (n + m - 1) * rho * w[n - 2, m]) / (n - m)

    ax = ay = az = 0.0
    for m in range(n_max + 1):
        for n in range(m, n_max + 1):
            cn, sn = cnm[n, m], snm[n, m]
            if m == 0:
                ax -= cn * v[n + 1, 1]
                ay -= cn * w[n + 1, 1]
                az -= (n + 1) * cn * v[n + 1, 0]
            else:
                fac = 0.5 * (n - m + 1) * (n - m + 2)
                ax += 0.5 * (-cn * v[n + 1, m + 1] - sn * w[n + 1, m + 1]) \
                    + fac * (cn * v[n + 1, m - 1] + sn * w[n + 1, m - 1])
                ay += 0.5 * (-cn * w[n + 1, m + 1] + sn * v[n + 1, m + 1]) \
                    + fac * (-cn * w[n + 1, m - 1] + sn * v[n + 1, m - 1])
                az += (n - m + 1) * (-cn * v[n + 1, m] - sn * w[n + 1, m])
    return gm / (r_ref * r_ref) * np.array([ax, ay, az])


def third_body_acceleration(r, r_body, gm_body):
    """Direct plus indirect perturbation in a Moon-centred frame.

    ``r`` is the spacecraft and ``r_body`` the perturbing body, both relative to the Moon.
    """
    d = r_body - r
    return gm_body * (d / np.linalg.norm(d) ** 3 - r_body / np.linalg.norm(r_body) ** 3)
