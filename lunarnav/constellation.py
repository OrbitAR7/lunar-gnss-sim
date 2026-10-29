"""Walker-type lunar navigation constellations and their propagation.

Two reference families, following the public LANS constellation trade studies
(Lunar Augmented Navigation Service, arXiv:2510.16030):

* ``s_elfo``   - Southern Elliptical Lunar Frozen Orbits: eccentricity set by the frozen
                 condition, argument of periapsis 90 deg so apolune dwells over the south.
* ``clfo``     - Circular Lunar Frozen Orbits at the lowest frozen inclination (~39.2 deg).

The numbers here are *example* designs, not any operator's baseline.
"""
from dataclasses import dataclass

import numpy as np

from .constants import DEG, SECONDS_PER_DAY
from .orbits import frozen_eccentricity, semi_major_axis_from_period
from .propagate import ForceModel, elements_to_mci, propagate
from .ephemeris import Ephemeris


@dataclass
class Constellation:
    name: str
    elements: list          # one (a, e, i, raan, argp, M) tuple per satellite, in the orbit-plane frame
    reference: str = "orbit_plane"

    @property
    def n_sat(self):
        return len(self.elements)


def walker(name, a, e, inc, argp, n_sats, n_planes, phasing, raan0=0.0):
    """Walker i:T/P/F pattern. RAAN spreads evenly over 360 deg across planes."""
    if n_sats % n_planes:
        raise ValueError("n_sats must be a multiple of n_planes")
    per_plane = n_sats // n_planes
    els = []
    for p in range(n_planes):
        raan = raan0 + 2 * np.pi * p / n_planes
        for k in range(per_plane):
            m = 2 * np.pi * k / per_plane + 2 * np.pi * p * phasing / n_sats
            els.append((a, e, inc, raan, argp, m % (2 * np.pi)))
    return Constellation(name, els)


def s_elfo(n_sats=12, n_planes=3, phasing=1, inc_deg=55.0, period_hours=24.0):
    inc = inc_deg * DEG
    a = semi_major_axis_from_period(period_hours * 3600.0)
    return walker(f"S-ELFO {inc_deg:g}:{n_sats}/{n_planes}/{phasing} ({period_hours:g} h)",
                  a, frozen_eccentricity(inc), inc, 90.0 * DEG, n_sats, n_planes, phasing)


def clfo(n_sats=24, n_planes=4, phasing=1, inc_deg=39.2, period_hours=12.0):
    inc = inc_deg * DEG
    a = semi_major_axis_from_period(period_hours * 3600.0)
    return walker(f"CLFO {inc_deg:g}:{n_sats}/{n_planes}/{phasing} ({period_hours:g} h)",
                  a, 0.0, inc, 0.0, n_sats, n_planes, phasing)


def propagate_constellation(const, jd0, duration_days, step_s, force=None):
    """MCI states for every satellite. Returns (t [s], states [n_sat, n_t, 6])."""
    force = force or ForceModel()
    t = np.arange(0.0, duration_days * SECONDS_PER_DAY + 0.5 * step_s, step_s)
    eph = Ephemeris(jd0, duration_days + 1.0)
    out = np.empty((const.n_sat, len(t), 6))
    for k, el in enumerate(const.elements):
        r0, v0 = elements_to_mci(el, jd0, const.reference)
        out[k] = propagate(r0, v0, jd0, t, force, eph)
    return t, out
