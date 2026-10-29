"""Keplerian element conversions and frozen-orbit design helpers."""
import numpy as np

from .constants import GM_MOON
from .frames import rot_x, rot_z


def solve_kepler(m, e, tol=1e-14):
    """Eccentric anomaly from mean anomaly (Newton iteration, safe for e < 0.99)."""
    m = np.mod(m, 2 * np.pi)
    ecc_anom = m if e < 0.8 else np.pi
    for _ in range(50):
        d = (ecc_anom - e * np.sin(ecc_anom) - m) / (1 - e * np.cos(ecc_anom))
        ecc_anom -= d
        if abs(d) < tol:
            break
    return ecc_anom


def kepler_to_cartesian(a, e, i, raan, argp, m, mu=GM_MOON):
    """Position/velocity [km, km/s] in the frame the angles are measured in."""
    ea = solve_kepler(m, e)
    cos_e, sin_e = np.cos(ea), np.sin(ea)
    r_pf = np.array([a * (cos_e - e), a * np.sqrt(1 - e * e) * sin_e, 0.0])
    r = np.linalg.norm(r_pf)
    v_pf = np.sqrt(mu * a) / r * np.array([-sin_e, np.sqrt(1 - e * e) * cos_e, 0.0])
    q = (rot_z(argp) @ rot_x(i) @ rot_z(raan)).T      # perifocal -> reference frame
    return q @ r_pf, q @ v_pf


def cartesian_to_kepler(r, v, mu=GM_MOON):
    """Return (a, e, i, raan, argp, M) in radians; the frame is that of ``r``/``v``."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    node = np.cross([0.0, 0.0, 1.0], h)
    nn = np.linalg.norm(node)
    ev = np.cross(v, h) / mu - r / rn
    e = np.linalg.norm(ev)
    a = 1.0 / (2.0 / rn - vn * vn / mu)
    i = np.arccos(h[2] / hn)
    raan = np.arctan2(node[1], node[0]) % (2 * np.pi) if nn > 1e-12 else 0.0
    if e > 1e-10 and nn > 1e-12:
        argp = np.arctan2(np.dot(np.cross(node, ev), h) / hn, np.dot(node, ev)) % (2 * np.pi)
    elif e > 1e-10:
        argp = np.arctan2(ev[1], ev[0]) % (2 * np.pi)
    else:
        argp = 0.0
    # true anomaly -> eccentric -> mean
    ref = ev if e > 1e-10 else (node if nn > 1e-12 else np.array([1.0, 0.0, 0.0]))
    nu = np.arctan2(np.dot(np.cross(ref, r), h) / hn, np.dot(ref, r))
    ea = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
    m = (ea - e * np.sin(ea)) % (2 * np.pi)
    return a, e, i, raan, argp, m


def period(a, mu=GM_MOON):
    return 2 * np.pi * np.sqrt(a**3 / mu)


def semi_major_axis_from_period(t_seconds, mu=GM_MOON):
    return (mu * (t_seconds / (2 * np.pi)) ** 2) ** (1.0 / 3.0)


def frozen_eccentricity(inc):
    """Eccentricity of the frozen lunar orbit at inclination ``inc`` [rad].

    Uses e^2 + (5/3) cos^2(i) = 1: the condition for zero secular apsidal drift at
    omega = 90 deg under the averaged Earth third-body (Kozai) perturbation, with J2
    neglected. It is the design relation quoted in public ELFO/LANS constellation studies
    (arXiv:2510.16030, Eq. 2). Because it comes from the third-body theory, the
    inclination must be measured from the Earth-Moon orbit plane, not the lunar equator
    (see ``R_icrf_to_orbit_plane``). tests/test_orbits.py and ``scripts/run_demo.py``
    measure how well it freezes the orbit under the full force model.
    """
    val = 1.0 - (5.0 / 3.0) * np.cos(inc) ** 2
    if val <= 0:
        raise ValueError("no frozen eccentricity: inclination below ~39.2 deg")
    return float(np.sqrt(val))
