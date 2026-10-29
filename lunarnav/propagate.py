"""Orbit propagation in the Moon-centred inertial (MCI, ICRF-aligned) frame."""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from . import constants as C
from .ephemeris import Ephemeris
from .frames import R_icrf_to_equator, R_icrf_to_orbit_plane, R_icrf_to_pa
from .gravity import default_coefficients, harmonic_acceleration, third_body_acceleration
from .orbits import cartesian_to_kepler, kepler_to_cartesian


@dataclass
class ForceModel:
    """Which perturbations are active. Harmonics are applied in the PA frame."""
    n_max: int = 3               # 0 = point mass, 2 = J2 + C22, 3 = adds degree 3
    earth: bool = True
    sun: bool = True
    zonal_only: bool = False     # keep only m = 0 terms (axisymmetric field)

    def coefficients(self):
        cnm, snm = default_coefficients(self.n_max)
        if self.zonal_only:
            cnm[:, 1:] = 0.0
            snm[:] = 0.0
        return cnm, snm


def propagate(r0, v0, jd0, t_eval, force=None, ephemeris=None, rtol=1e-11, atol=1e-9,
              pa_rotation=R_icrf_to_pa):
    """Integrate an MCI state from ``jd0``; returns states [km, km/s], shape (len(t_eval), 6).

    ``t_eval`` are seconds after ``jd0``. The ephemeris must cover the span.
    ``pa_rotation(jd)`` returns the ICRF -> PA matrix (default: IAU series + constant offset).
    """
    force = force or ForceModel()
    t_eval = np.asarray(t_eval, float)
    cnm, snm = force.coefficients()
    if ephemeris is None and (force.earth or force.sun):
        ephemeris = Ephemeris(jd0, t_eval.max() / 86400.0 + 1.0)

    def rhs(t, y):
        r = y[:3]
        r_pa = pa_rotation(jd0 + t / C.SECONDS_PER_DAY)
        a = r_pa.T @ harmonic_acceleration(r_pa @ r, cnm, snm)
        if force.earth:
            a += third_body_acceleration(r, ephemeris.earth_wrt_moon(t), C.GM_EARTH)
        if force.sun:
            a += third_body_acceleration(r, ephemeris.sun_wrt_moon(t), C.GM_SUN)
        return np.concatenate([y[3:], a])

    sol = solve_ivp(rhs, (0.0, t_eval[-1]), np.concatenate([r0, v0]), method="DOP853",
                    t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


_REFERENCE_FRAMES = {"orbit_plane": R_icrf_to_orbit_plane, "equator": R_icrf_to_equator}


def elements_to_mci(elements, jd0, reference="orbit_plane"):
    """Elements (a, e, i, raan, argp, M) in a reference frame at ``jd0`` -> MCI state.

    ``reference`` is "orbit_plane" (inclination measured from the mean Earth-Moon orbit
    plane, the convention frozen-orbit theory needs) or "equator" (lunar equator of date).
    """
    r_ref, v_ref = kepler_to_cartesian(*elements)
    q = _REFERENCE_FRAMES[reference](jd0).T
    return q @ r_ref, q @ v_ref


def mci_to_elements(states, jds, reference="orbit_plane"):
    """Osculating elements of MCI ``states`` in the chosen reference frame *of date*.

    Re-evaluating the frame at each epoch keeps inclination meaningful while the plane
    regresses; the frame rate (~1e-9 rad/s) is ignored in the velocity transform.
    """
    frame = _REFERENCE_FRAMES[reference]
    out = np.empty((len(jds), 6))
    for k, (s, jd) in enumerate(zip(states, jds)):
        q = frame(jd)
        out[k] = cartesian_to_kepler(q @ s[:3], q @ s[3:])
    return out
