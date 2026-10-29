#!/usr/bin/env python3
"""End-to-end demo: lunar frames -> frozen orbits -> two GNSS-like constellations -> DOP.

    python scripts/run_demo.py            # ~3-4 min, 28 days, 800 users
    python scripts/run_demo.py --quick    # ~1 min, 7 days, 200 users

Writes figures to docs/figures/ and numbers to results/results.json + docs/results.md.
"""
import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lunarnav import constants as C
from lunarnav.constellation import clfo, propagate_constellation, s_elfo
from lunarnav.ephemeris import moon_geocentric
from lunarnav.frames import (R_icrf_to_me, R_icrf_to_orbit_plane, R_pa_to_me, enu_basis,
                             latlon_to_unit, unit_to_latlon)
from lunarnav.gnss import (dop_series, fibonacci_sphere, satellites_in_me, single_point_solution)
from lunarnav.orbits import frozen_eccentricity
from lunarnav.propagate import ForceModel, elements_to_mci, mci_to_elements, propagate
from lunarnav.timeutil import jd_tt_from_utc

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
JD0 = jd_tt_from_utc(datetime(2027, 1, 1, tzinfo=timezone.utc))
DEG = np.pi / 180

# Reference data-viz palette (categorical slots 1-3) and light surface.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 130,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", FIG / name)


# --------------------------------------------------------------------------- frames
def fig_frames():
    jds = JD0 + np.arange(0, 2 * 365.25, 0.125)
    lon, lat = np.empty(len(jds)), np.empty(len(jds))
    for k, jd in enumerate(jds):
        la, lo = unit_to_latlon(R_icrf_to_me(jd) @ (-moon_geocentric(jd)))
        lat[k], lon[k] = la, lo
    lon, lat = np.degrees(lon), np.degrees(lat)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.3})
    a.plot(lon, lat, lw=0.35, color=BLUE, alpha=0.8)
    a.plot(0, 0, "+", color=ORANGE, ms=14, mew=2, label="mean Earth direction (ME +X)")
    a.set_xlabel("sub-Earth selenographic longitude [deg]")
    a.set_ylabel("sub-Earth latitude [deg]")
    a.set_title("Earth libration in the ME frame (2 yr)")
    a.set_aspect("equal")
    a.set_xlim(-10, 10); a.set_ylim(-9, 9)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=2,
             title=f"max |lon| {np.abs(lon).max():.1f}°, max |lat| {np.abs(lat).max():.1f}°",
             title_fontsize=8)

    # A point's PA coordinates u, misread as ME coordinates, land at u in ME; the point is really
    # at rot @ u. Arrows go from the misread location to the true one, in local east/north metres.
    rot = R_pa_to_me()                      # v_ME = rot @ v_PA
    la_g, lo_g = np.meshgrid(np.radians(np.arange(-60, 61, 30)), np.radians(np.arange(-150, 151, 30)))
    la_g, lo_g = la_g.ravel(), lo_g.ravel()
    u = latlon_to_unit(la_g, lo_g)
    d = (u @ rot.T - u) * C.R_MEAN * 1e3    # metres, ME components
    east = np.array([enu_basis(x, y)[0] @ dd for x, y, dd in zip(la_g, lo_g, d)])
    north = np.array([enu_basis(x, y)[1] @ dd for x, y, dd in zip(la_g, lo_g, d)])
    # angles="uv": arrow direction is the true local compass direction, not distorted by
    # the unequal degree scales of the map axes.
    q = b.quiver(np.degrees(lo_g), np.degrees(la_g), east, north, color=BLUE, angles="uv",
                 scale_units="inches", scale=2400.0, width=0.0045, pivot="middle")
    b.quiverkey(q, 0.82, -0.17, 500, "500 m", labelpos="E", coordinates="axes", fontproperties={"size": 8})
    b.set_xlim(-180, 180)
    b.set_ylim(-80, 80)
    b.set_xlabel("selenographic longitude [deg]")
    b.set_ylabel("latitude [deg]")
    angle = np.degrees(np.arccos((np.trace(rot) - 1) / 2))
    b.set_title(f"PA vs ME (DE440): {angle:.4f}°, up to {np.hypot(east, north).max():.0f} m")
    save(fig, "frames.png")
    return dict(lon_lib_max_deg=float(np.abs(lon).max()), lat_lib_max_deg=float(np.abs(lat).max()),
                lon_mean_deg=float(lon.mean()), lat_mean_deg=float(lat.mean()),
                pa_me_angle_deg=float(np.degrees(np.arccos((np.trace(rot) - 1) / 2))),
                pa_me_max_shift_m=float(np.hypot(east, north).max()))


# --------------------------------------------------------------------------- frozen orbit
def fig_frozen(days=90):
    a = 9750.734
    inc, e0 = 55 * DEG, frozen_eccentricity(55 * DEG)
    t = np.arange(0.0, days * 86400.0, 3600.0)
    out = {}
    for label, ref in (("orbit-plane reference (design)", "orbit_plane"), ("lunar-equator reference (naive)", "equator")):
        r0, v0 = elements_to_mci((a, e0, inc, 30 * DEG, 90 * DEG, 0.0), JD0, ref)
        st = propagate(r0, v0, JD0, t, ForceModel())
        jds = np.full(len(t), JD0) if ref == "equator" else JD0 + t / 86400.0
        out[label] = mci_to_elements(st, jds, ref)
    fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for (label, el), col in zip(out.items(), (BLUE, ORANGE)):
        d = t / 86400.0
        axs[0].plot(d, el[:, 1], color=col, lw=1.4, label=label)
        axs[1].plot(d, np.degrees(el[:, 2]), color=col, lw=1.4)
        axs[2].plot(d, np.degrees(np.unwrap(el[:, 4])), color=col, lw=1.4)
    axs[0].set_ylabel("eccentricity")
    axs[1].set_ylabel("inclination [deg]")
    axs[2].set_ylabel("arg. of periapsis [deg]")
    axs[2].set_xlabel("days since 2027-01-01")
    axs[0].legend(fontsize=8.5, loc="lower left", ncol=2, bbox_to_anchor=(0.0, 1.0))
    axs[0].set_title("24 h ELFO, i = 55°, e = frozen value, no station-keeping", pad=24)
    save(fig, "frozen_orbit.png")
    summ = {}
    for label, el in out.items():
        w = np.degrees(np.unwrap(el[:, 4]))
        summ[label] = dict(e_range=[float(el[:, 1].min()), float(el[:, 1].max())],
                           inc_range_deg=[float(np.degrees(el[:, 2]).min()), float(np.degrees(el[:, 2]).max())],
                           argp_slope_deg_per_day=float(np.polyfit(t / 86400, w, 1)[0]),
                           argp_range_deg=float(np.ptp(w)))
    return summ


# --------------------------------------------------------------------------- constellations
def region_stats(n_vis, dops, sel):
    nv, p, h = n_vis[sel], dops["pdop"][sel], dops["hdop"][sel]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slice (never 4 sats) -> NaN
        median_pdop, p95 = float(np.nanmedian(p)), float(np.nanpercentile(p, 95))
        median_hdop = float(np.nanmedian(h))
    return dict(
        users=int(sel.sum()),
        avail_4_sats_pct=float(100 * (nv >= 4).mean()),
        pdop_le_6_pct=float(100 * (p <= 6).mean()),
        pdop_le_3_pct=float(100 * (p <= 3).mean()),
        median_pdop=median_pdop,
        p95_pdop=p95,
        median_hdop=median_hdop,
        mean_visible=float(nv.mean()),
        min_visible=int(nv.min()),
    )


def run_constellation(const, days, step_s, users_unit, named):
    t0 = time.time()
    t, states = propagate_constellation(const, JD0, days, step_s)
    jds = JD0 + t / 86400.0
    sats = satellites_in_me(states, jds)
    n_vis, dops = dop_series(np.vstack([users_unit, named]), sats, mask_deg=5.0)
    print(f"  {const.name}: {const.n_sat} sats, {len(t)} epochs, {time.time() - t0:.0f} s")
    return t, states, sats, n_vis, dops


def fig_coverage(users_unit, results):
    lat, lon = np.degrees(unit_to_latlon(users_unit))
    n = len(users_unit)
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.2))
    for row, (name, res) in enumerate(results.items()):
        _, _, _, n_vis, dops = res
        pd6 = 100 * (np.nan_to_num(dops["pdop"][:n], nan=1e9) <= 6).mean(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(dops["pdop"][:n], axis=1)
        ax = axs[row, 0]
        s = ax.scatter(lon, lat, c=pd6, s=16, cmap="Blues", vmin=-15, vmax=100, edgecolors=INK2, linewidths=0.3)
        ax.set_title(f"{name}\n% of time with PDOP ≤ 6")
        ax.set_xlabel("longitude [deg]"); ax.set_ylabel("latitude [deg]")
        fig.colorbar(s, ax=ax, pad=0.01)
        ax = axs[row, 1]
        s = ax.scatter(lon, lat, c=np.clip(np.nan_to_num(med, nan=20.0), 0, 20), s=16, cmap="Blues_r",
                       vmin=0, vmax=35, edgecolors=INK2, linewidths=0.3)
        ax.set_title("median PDOP (no fix at all = 20)")
        ax.set_xlabel("longitude [deg]")
        fig.colorbar(s, ax=ax, pad=0.01)
    fig.tight_layout()
    save(fig, "coverage_maps.png")


def fig_southpole(results, named_index, days_shown=3):
    fig, axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for (name, res), col in zip(results.items(), (BLUE, ORANGE)):
        t, _, _, n_vis, dops = res
        k = named_index["south pole"]
        m = t <= days_shown * 86400
        axs[0].step(t[m] / 86400, n_vis[k][m], where="post", color=col, lw=1.3, label=name)
        axs[1].plot(t[m] / 86400, dops["pdop"][k][m], color=col, lw=1.3)
    axs[0].axhline(4, color=INK2, lw=0.8, ls=":")
    axs[0].set_ylabel("visible satellites (5° mask)")
    axs[1].set_ylabel("PDOP")
    axs[1].set_yscale("log")
    axs[1].set_yticks([2, 3, 5, 10, 20]); axs[1].set_yticklabels(["2", "3", "5", "10", "20"])
    axs[1].minorticks_off()
    axs[0].set_ylim(3.5, 12.5)
    axs[1].set_xlabel("days since 2027-01-01")
    axs[0].legend(fontsize=8.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    axs[0].set_title("User at the lunar south pole (-89.9°)", pad=14)
    save(fig, "south_pole_timeseries.png")


def fig_orbits(consts):
    fig = plt.figure(figsize=(11.5, 8))
    q0 = R_icrf_to_orbit_plane(JD0)
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, np.pi, 20)
    sx, sy, sz = (C.R_MEAN * np.outer(np.cos(u), np.sin(v)), C.R_MEAN * np.outer(np.sin(u), np.sin(v)),
                  C.R_MEAN * np.outer(np.ones_like(u), np.cos(v)))
    earth_dir = q0 @ (-moon_geocentric(JD0))
    earth_dir /= np.linalg.norm(earth_dir)
    # The analysis grid (10 min) is too coarse to draw: near a 24 h ELFO's perilune the satellite
    # sweeps ~17 deg per sample, so tracks would be drawn as straight chords. Re-propagate the
    # drawn satellites on a 1 min grid instead (same force model and initial states).
    t_draw = np.arange(0.0, 3 * 86400.0 + 1.0, 60.0)
    for k, (const, col) in enumerate(zip(consts, (BLUE, ORANGE))):
        n_planes = len({round(el[3], 9) for el in const.elements})
        per_plane = const.n_sat // n_planes
        drawn = []
        for el in const.elements[::per_plane]:          # one satellite per plane
            r0, v0 = elements_to_mci(el, JD0, const.reference)
            drawn.append(propagate(r0, v0, JD0, t_draw, ForceModel()))
        ax = fig.add_subplot(2, 2, 1 + k, projection="3d")
        ax.plot_surface(sx, sy, sz, color="#cfcdc6", alpha=0.9, linewidth=0)
        one_day = t_draw <= 86400.0
        xyz_all = [(q0 @ s[one_day, :3].T).T for s in drawn]
        for xyz in xyz_all:
            ax.plot(*xyz.T, color=col, lw=0.9)
        lim = np.abs(np.array(xyz_all)).max()
        ax.quiver(0, 0, 0, *(earth_dir * lim * 0.9), color=INK, lw=1.5, arrow_length_ratio=0.08)
        ax.text(*(earth_dir * lim * 0.95), "Earth", fontsize=8)
        ax.set_title(f"{const.name}\ninertial, orbit-plane frame (1 day)")
        ax.set_box_aspect((1, 1, 1))
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(-lim, lim)
        ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("z [km]")
        ax.locator_params(nbins=4)
        ax.view_init(elev=18, azim=-60)

        ax = fig.add_subplot(2, 2, 3 + k)
        track_me = satellites_in_me(drawn[0][None, :, :3], JD0 + t_draw / 86400.0)[0]
        lat, lon = unit_to_latlon(track_me)
        lon_d, lat_d = np.degrees(lon), np.degrees(lat)
        jump = np.where(np.abs(np.diff(lon_d)) > 180)[0] + 1
        lon_p, lat_p = np.insert(lon_d, jump, np.nan), np.insert(lat_d, jump, np.nan)
        ax.plot(lon_p, lat_p, color=col, lw=1.0)
        hourly = slice(0, None, 60)
        ax.plot(lon_d[hourly], lat_d[hourly], "o", ms=2.5, color=col, label="hourly position")
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2)
        ax.set_title("sub-satellite track of satellite 1 in ME (3 days)", pad=20)
        ax.set_xlabel("longitude [deg]"); ax.set_ylabel("latitude [deg]")
    fig.tight_layout()
    save(fig, "constellation_orbits.png")


# --------------------------------------------------------------------------- SPP check
def fig_spp(users_all, res, n_users, rng, n_samples=250, draws=150, sigma_km=0.001):
    _, _, sats, n_vis, dops = res
    cand = np.argwhere((n_vis[:n_users] >= 5) & (dops["pdop"][:n_users] < 25))
    pick = cand[rng.choice(len(cand), size=min(n_samples, len(cand)), replace=False)]
    pdop, rms = [], []
    for ui, ti in pick:
        user = users_all[ui] * C.R_MEAN
        s = sats[:, ti, :]
        vis = (((s - user) @ users_all[ui]) / np.linalg.norm(s - user, axis=1)) >= np.sin(5 * DEG)
        sv = s[vis]
        pr = np.linalg.norm(sv - user, axis=1)
        errs = [np.linalg.norm(single_point_solution(sv, pr + rng.normal(0, sigma_km, len(sv)),
                                                      x0=np.append(user, 0.0))[:3] - user) for _ in range(draws)]
        pdop.append(dops["pdop"][ui, ti])
        rms.append(np.sqrt(np.mean(np.square(errs))) / sigma_km)
    pdop, rms = np.array(pdop), np.array(rms)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(pdop, rms, s=14, color=BLUE, alpha=0.7)
    lim = [0, max(pdop.max(), rms.max()) * 1.05]
    ax.plot(lim, lim, color=ORANGE, lw=1.5, label="theory: RMS error = PDOP × σ")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("PDOP from geometry"); ax.set_ylabel("simulated 3-D RMS error / σ (UERE)")
    ax.set_title("Single-point solutions vs DOP")
    ax.legend(fontsize=8.5)
    save(fig, "spp_vs_pdop.png")
    return dict(samples=len(pdop), median_ratio=float(np.median(rms / pdop)),
                std_ratio_pct=float(100 * np.std(rms / pdop)))


# --------------------------------------------------------------------------- main
def markdown(res):
    lines = ["# Results (generated by `scripts/run_demo.py`)", ""]
    lines += [f"Run: {res['meta']['days']} days from 2027-01-01, {res['meta']['step_s']:.0f} s step, "
              f"{res['meta']['users']} Fibonacci users + named sites, 5° elevation mask, "
              "full force model (degree-3 lunar field, Earth + Sun).", ""]
    f = res["frames"]
    lines += ["## Frames", "",
              f"- sub-Earth libration envelope: ±{f['lon_lib_max_deg']:.1f}° lon, ±{f['lat_lib_max_deg']:.1f}° lat; "
              f"mean ({f['lon_mean_deg']:+.2f}°, {f['lat_mean_deg']:+.2f}°)",
              f"- PA↔ME rotation {f['pa_me_angle_deg']:.5f}° → up to {f['pa_me_max_shift_m']:.0f} m on the surface", ""]
    lines += ["## Frozen-orbit check (90 d, no station-keeping)", "",
              "| reference plane | e range | i range [deg] | ω drift [deg/day] |", "|---|---|---|---|"]
    for k, v in res["frozen"].items():
        lines.append(f"| {k} | {v['e_range'][0]:.3f}–{v['e_range'][1]:.3f} | "
                     f"{v['inc_range_deg'][0]:.1f}–{v['inc_range_deg'][1]:.1f} | {v['argp_slope_deg_per_day']:+.3f} |")
    lines += ["", "## Navigation performance", "",
              "| constellation | region | 4+ sats [%] | PDOP≤6 [%] | PDOP≤3 [%] | median PDOP | 95th PDOP | median HDOP |",
              "|---|---|---|---|---|---|---|---|"]
    for cname, regs in res["performance"].items():
        for rname, s in regs.items():
            lines.append(f"| {cname} | {rname} | {s['avail_4_sats_pct']:.1f} | {s['pdop_le_6_pct']:.1f} | "
                         f"{s['pdop_le_3_pct']:.1f} | {s['median_pdop']:.2f} | {s['p95_pdop']:.1f} | {s['median_hdop']:.2f} |")
    lines += ["", "## Single-point-solution consistency", "",
              f"{res['spp']['samples']} user/epoch samples: simulated RMS error ÷ (PDOP·σ) has median "
              f"{res['spp']['median_ratio']:.3f} and scatter {res['spp']['std_ratio_pct']:.1f} % "
              "(sampling noise alone with 150 draws is ~3.3 %; the excess is not investigated).", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=28.0)
    ap.add_argument("--users", type=int, default=800)
    ap.add_argument("--step-min", type=float, default=10.0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.days, args.users = 7.0, 200
    step_s = args.step_min * 60.0
    rng = np.random.default_rng(2027)

    print("frames");        frames = fig_frames()
    print("frozen orbit");  frozen = fig_frozen()

    users = fibonacci_sphere(args.users)
    sites = {"south pole": (-89.9, 0.0), "Malapert region": (-86.0, 0.1),
             "near-side equator": (0.0, 0.0), "far-side equator": (0.0, 180.0), "north pole": (89.9, 0.0)}
    named = np.array([latlon_to_unit(np.radians(la), np.radians(lo)) for la, lo in sites.values()])
    named_index = {k: args.users + i for i, k in enumerate(sites)}

    consts = [s_elfo(12, 3, 1, 55.0, 24.0), clfo(24, 4, 1, 39.2, 12.0)]
    print("constellations")
    results = {c.name: run_constellation(c, args.days, step_s, users, named) for c in consts}

    lat_all = np.degrees(unit_to_latlon(np.vstack([users, named]))[0])
    performance = {}
    for c in consts:
        _, _, _, n_vis, dops = results[c.name]
        grid = np.arange(len(lat_all)) < args.users
        perf = {"global": region_stats(n_vis, dops, grid),
                "south polar cap (lat <= -80)": region_stats(n_vis, dops, grid & (lat_all <= -80))}
        for site, idx in named_index.items():
            perf[site] = region_stats(n_vis, dops, np.arange(len(lat_all)) == idx)
        performance[c.name] = perf

    fig_coverage(users, results)
    fig_southpole(results, named_index)
    fig_orbits(consts)
    spp = fig_spp(np.vstack([users, named]), results[consts[0].name], args.users, rng)

    out = dict(meta=dict(days=args.days, step_s=step_s, users=args.users), frames=frames, frozen=frozen,
               performance=performance, spp=spp)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "results.json").write_text(json.dumps(out, indent=2))
    (ROOT / "docs" / "results.md").write_text(markdown(out))
    print("done ->", ROOT / "docs" / "results.md")


if __name__ == "__main__":
    main()
