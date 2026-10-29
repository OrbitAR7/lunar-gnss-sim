"""Minimal time handling. Epochs are Julian dates in TDB, treated as TT.

TDB - TT is a periodic term of at most ~1.7 ms; at lunar orbital speeds (~1.5 km/s)
that is a few metres of along-track error, far below the accuracy of the analytic
ephemerides used here, so it is ignored.
"""
from datetime import datetime, timezone

import numpy as np

from .constants import (JD_J2000, SECONDS_PER_DAY, TAI_MINUS_UTC, TT_MINUS_TAI)


def jd_utc(dt: datetime) -> float:
    """Julian date (UTC) of a timezone-aware datetime."""
    dt = dt.astimezone(timezone.utc)
    y, m = dt.year, dt.month
    if m <= 2:
        y, m = y - 1, m + 12
    a = y // 100
    b = 2 - a + a // 4
    jd0 = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + dt.day + b - 1524.5
    frac = (dt.hour + (dt.minute + (dt.second + dt.microsecond * 1e-6) / 60.0) / 60.0) / 24.0
    return jd0 + frac


def jd_tt_from_utc(dt: datetime) -> float:
    """TT Julian date from UTC. Only valid on or after 2017-01-01 (TAI-UTC = 37 s)."""
    if dt.astimezone(timezone.utc) < datetime(2017, 1, 1, tzinfo=timezone.utc):
        raise ValueError("fixed leap-second count is only valid from 2017-01-01")
    return jd_utc(dt) + (TAI_MINUS_UTC + TT_MINUS_TAI) / SECONDS_PER_DAY


def days_since_j2000(jd):
    return np.asarray(jd) - JD_J2000


def centuries_since_j2000(jd):
    return (np.asarray(jd) - JD_J2000) / 36525.0
