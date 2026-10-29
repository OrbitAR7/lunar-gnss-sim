"""Visibility, dilution of precision and single-point positioning for a lunar GNSS.

Users sit on a spherical Moon of radius ``R_MEAN`` (no terrain, no Earth/Sun eclipse of
signals) in the ME body-fixed frame. A satellite is usable if it is above the elevation
mask. Pseudoranges are geometric range plus a common receiver clock bias; the DOP
matrix is the standard (H^T H)^-1 with H rows [-line-of-sight, 1].
"""
import numpy as np

from .constants import R_MEAN
from .frames import R_icrf_to_me, enu_basis, unit_to_latlon


def fibonacci_sphere(n):
    """~Uniformly spread unit vectors (golden-angle spiral)."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    phi = np.pi * (1 + 5 ** 0.5) * k
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def satellites_in_me(states_mci, jds):
    """Rotate MCI positions [n_sat, n_t, >=3] into the ME frame at each epoch."""
    rot = np.array([R_icrf_to_me(jd) for jd in jds])               # (n_t, 3, 3)
    return np.einsum("tij,stj->sti", rot, states_mci[..., :3])


def _dop_chunk(users, sats, mask_rad, radius):
    """users (U,3 unit), sats (S,T,3 km, ME) -> n_vis, dop arrays of shape (U,T)."""
    pos = users * radius                                           # (U,3)
    rho = sats[None, :, :, :] - pos[:, None, None, :]              # (U,S,T,3)
    rng = np.linalg.norm(rho, axis=-1)
    los = rho / rng[..., None]
    sin_el = np.einsum("ustk,uk->ust", los, users)
    vis = sin_el >= np.sin(mask_rad)                               # (U,S,T)
    n_vis = vis.sum(axis=1)

    h = np.concatenate([-los, np.ones_like(rng)[..., None]], axis=-1)   # (U,S,T,4)
    h = h * vis[..., None]
    hth = np.einsum("ustp,ustq->utpq", h, h)                       # (U,T,4,4)
    ok = n_vis >= 4
    hth[~ok] = np.eye(4)
    q = np.linalg.inv(hth)
    q[~ok] = np.nan

    lat, lon = unit_to_latlon(users)
    enu = np.array([enu_basis(la, lo) for la, lo in zip(lat, lon)])     # (U,3,3)
    q_enu = np.einsum("uij,utjk,ulk->util", enu, q[..., :3, :3], enu)
    pdop = np.sqrt(np.trace(q[..., :3, :3], axis1=-2, axis2=-1))
    hdop = np.sqrt(q_enu[..., 0, 0] + q_enu[..., 1, 1])
    vdop = np.sqrt(q_enu[..., 2, 2])
    tdop = np.sqrt(q[..., 3, 3])
    gdop = np.sqrt(pdop**2 + tdop**2)
    return n_vis, dict(pdop=pdop, hdop=hdop, vdop=vdop, tdop=tdop, gdop=gdop)


def dop_series(users, sats_me, mask_deg=5.0, radius=R_MEAN, chunk_bytes=150e6):
    """Visible-satellite count and DOPs for every user and epoch.

    ``users``: (U,3) unit vectors in ME; ``sats_me``: (S,T,3) km. Returns ``n_vis`` (U,T)
    and a dict of DOP arrays (U,T); DOPs are NaN where fewer than 4 satellites are visible.
    """
    users = np.atleast_2d(users)
    n_s, n_t, _ = sats_me.shape
    per_user = n_s * n_t * (3 * 8 * 4)
    step = max(1, int(chunk_bytes // per_user))
    n_vis = np.empty((len(users), n_t), dtype=int)
    dops = {k: np.empty((len(users), n_t)) for k in ("pdop", "hdop", "vdop", "tdop", "gdop")}
    for lo in range(0, len(users), step):
        sl = slice(lo, lo + step)
        nv, dp = _dop_chunk(users[sl], sats_me, np.radians(mask_deg), radius)
        n_vis[sl] = nv
        for k in dops:
            dops[k][sl] = dp[k]
    return n_vis, dops


def single_point_solution(sat_pos, pseudoranges, x0=None, iterations=15):
    """Gauss-Newton position + clock-bias fix. Units: km. Returns [x, y, z, c*dt]."""
    x = np.zeros(4) if x0 is None else np.array(x0, float)
    for _ in range(iterations):
        diff = x[:3] - sat_pos
        rng = np.linalg.norm(diff, axis=1)
        h = np.hstack([diff / rng[:, None], np.ones((len(rng), 1))])
        dx, *_ = np.linalg.lstsq(h, pseudoranges - (rng + x[3]), rcond=None)
        x += dx
        if np.linalg.norm(dx) < 1e-12:
            break
    return x
