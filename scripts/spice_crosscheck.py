#!/usr/bin/env python3
"""Cross-check lunarnav against NAIF SPICE (DE440s + moon_pa_de440 + lunar frame kernel).

Needs spiceypy and the kernels in ./kernels (see README). Run isolated, since the
kernels are downloaded data:  python -I scripts/spice_crosscheck.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import spiceypy as sp
from scipy.interpolate import CubicSpline

from lunarnav import constants as C
from lunarnav.constellation import s_elfo
from lunarnav.ephemeris import Ephemeris, moon_geocentric, sun_geocentric
from lunarnav.frames import R_icrf_to_me, R_icrf_to_pa, R_pa_to_me
from lunarnav.propagate import ForceModel, elements_to_mci, propagate
from lunarnav.timeutil import jd_tt_from_utc

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "kernels"
for f in ("naif0012.tls", "de440s.bsp", "pck00011.tpc", "moon_pa_de440_200625.bpc", "moon_de440_250416.tf"):
    sp.furnsh(str(K / f))

JD0 = jd_tt_from_utc(datetime(2027, 1, 1, tzinfo=timezone.utc))
et_of = lambda jd: (jd - C.JD_J2000) * 86400.0          # TDB seconds; our TT is treated as TDB
ang = lambda ra, rb: np.arccos(np.clip((np.trace(ra @ rb.T) - 1) / 2, -1, 1))
out = {}

# 1. ephemeris -------------------------------------------------------------
jds = JD0 + np.arange(0, 3 * 365, 1.0)
dm, dang, dsun = [], [], []
for jd in jds:
    ref = sp.spkpos("MOON", et_of(jd), "J2000", "NONE", "EARTH")[0]
    mine = moon_geocentric(jd)
    dm.append(np.linalg.norm(mine - ref))
    dang.append(np.degrees(np.arccos(mine @ ref / np.linalg.norm(mine) / np.linalg.norm(ref))))
    sref = sp.spkpos("SUN", et_of(jd), "J2000", "NONE", "EARTH")[0]
    dsun.append(np.degrees(np.arccos(sun_geocentric(jd) @ sref / np.linalg.norm(sref) / np.linalg.norm(sun_geocentric(jd)))))
out["moon_ephemeris"] = dict(rms_km=float(np.sqrt(np.mean(np.square(dm)))), max_km=float(np.max(dm)),
                             max_dir_deg=float(np.max(dang)), sun_max_dir_deg=float(np.max(dsun)))

# 2. frames ----------------------------------------------------------------
jds = JD0 + np.arange(0, 19 * 365.25, 5.0)
d_iau, d_me, d_pa = [], [], []
for jd in jds:
    et = et_of(jd)
    d_iau.append(ang(R_icrf_to_me(jd), sp.pxform("J2000", "IAU_MOON", et)))
    d_me.append(ang(R_icrf_to_me(jd), sp.pxform("J2000", "MOON_ME", et)))
    d_pa.append(ang(R_icrf_to_pa(jd), sp.pxform("J2000", "MOON_PA", et)))
m = lambda a: float(np.max(a) * C.R_MEAN * 1e3)
out["frames"] = dict(
    mine_vs_spice_IAU_MOON_max_m=m(d_iau),
    iau_series_vs_de440_ME_max_m=m(d_me), iau_series_vs_de440_ME_rms_m=float(np.sqrt(np.mean(np.square(d_me))) * C.R_MEAN * 1e3),
    mine_PA_vs_de440_PA_max_m=m(d_pa))

# 3. PA -> ME constant offset (sign check) ----------------------------------
r_spice = sp.pxform("MOON_PA", "MOON_ME", et_of(JD0))
out["pa_me_offset"] = dict(
    my_matrix_vs_spice_max_abs_elem=float(np.max(np.abs(R_pa_to_me() - r_spice))),
    transposed_vs_spice_max_abs_elem=float(np.max(np.abs(R_pa_to_me().T - r_spice))),
    shift_if_sign_wrong_m=float(2 * np.arccos((np.trace(R_pa_to_me()) - 1) / 2) * C.R_MEAN * 1e3))

# 4. Earth libration with DE440 ME ------------------------------------------
lat, lon = [], []
for jd in JD0 + np.arange(0, 19 * 365.25, 0.5):
    e = sp.mxv(sp.pxform("J2000", "MOON_ME", et_of(jd)), sp.spkpos("EARTH", et_of(jd), "J2000", "NONE", "MOON")[0])
    r, lo, la = sp.reclat(e)
    lat.append(np.degrees(la)); lon.append(np.degrees(lo))
out["de440_libration"] = dict(lon_mean=float(np.mean(lon)), lat_mean=float(np.mean(lat)),
                              lon_max=float(np.max(np.abs(lon))), lat_max=float(np.max(np.abs(lat))))

# 5. effect on a propagated orbit: analytic vs SPICE ephemeris + PA frame ----
class SpiceEphemeris:
    def __init__(self, jd0, days, step_min=30.0):
        secs = np.arange(-step_min * 60, days * 86400 + 3 * step_min * 60, step_min * 60.0)
        ets = et_of(jd0) + secs
        self._e = CubicSpline(secs, np.array([sp.spkpos("EARTH", t, "J2000", "NONE", "MOON")[0] for t in ets]))
        self._s = CubicSpline(secs, np.array([sp.spkpos("SUN", t, "J2000", "NONE", "MOON")[0] for t in ets]))

    earth_wrt_moon = lambda self, t: self._e(t)
    sun_wrt_moon = lambda self, t: self._s(t)


spice_pa = lambda jd: np.array(sp.pxform("J2000", "MOON_PA", et_of(jd)))
const = s_elfo(3, 3)
t = np.arange(0.0, 30 * 86400.0, 3600.0)
r0, v0 = elements_to_mci(const.elements[0], JD0, const.reference)
mine = propagate(r0, v0, JD0, t, ForceModel(), Ephemeris(JD0, 31.0))
ref = propagate(r0, v0, JD0, t, ForceModel(), SpiceEphemeris(JD0, 31.0), pa_rotation=spice_pa)
dpos = np.linalg.norm(mine[:, :3] - ref[:, :3], axis=1)
out["orbit_30d_analytic_vs_spice"] = dict(max_position_diff_km=float(dpos.max()), at_day_10_km=float(dpos[240]),
                                          at_day_30_km=float(dpos[-1]))
print(json.dumps(out, indent=2))
(ROOT / "results" / "spice_crosscheck.json").write_text(json.dumps(out, indent=2))
