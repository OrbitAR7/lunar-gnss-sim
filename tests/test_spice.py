"""Cross-checks against NAIF SPICE. Skipped unless spiceypy and ./kernels are present."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sp = pytest.importorskip("spiceypy")

from lunarnav import constants as C
from lunarnav.ephemeris import moon_geocentric
from lunarnav.frames import R_icrf_to_me, R_icrf_to_pa, R_pa_to_me
from lunarnav.timeutil import jd_tt_from_utc

K = Path(__file__).resolve().parents[1] / "kernels"
FILES = ("naif0012.tls", "de440s.bsp", "pck00011.tpc", "moon_pa_de440_200625.bpc", "moon_de440_250416.tf")
pytestmark = pytest.mark.skipif(not all((K / f).exists() for f in FILES), reason="NAIF kernels not downloaded")

JD0 = jd_tt_from_utc(datetime(2027, 1, 1, tzinfo=timezone.utc))
et = lambda jd: (jd - C.JD_J2000) * 86400.0
angle = lambda a, b: np.arccos(np.clip((np.trace(a @ b.T) - 1) / 2, -1, 1))


@pytest.fixture(scope="module", autouse=True)
def kernels():
    for f in FILES:
        sp.furnsh(str(K / f))
    yield
    sp.kclear()


def test_pa_to_me_offset_matches_spice_including_sign():
    np.testing.assert_allclose(R_pa_to_me(), sp.pxform("MOON_PA", "MOON_ME", et(JD0)), atol=1e-15)


def test_iau_series_matches_spice_iau_moon():
    for d in (0.0, 700.0, 4000.0):
        err = angle(R_icrf_to_me(JD0 + d), np.array(sp.pxform("J2000", "IAU_MOON", et(JD0 + d))))
        assert err * C.R_MEAN * 1e3 < 1.0                      # same model: agree to < 1 m


def test_iau_series_vs_de440_is_within_published_150_m():
    errs = [angle(R_icrf_to_pa(JD0 + d), np.array(sp.pxform("J2000", "MOON_PA", et(JD0 + d))))
            for d in np.arange(0.0, 6000.0, 37.0)]
    assert max(errs) * C.R_MEAN * 1e3 < 160.0


def test_analytic_moon_vs_de440():
    for d in np.arange(0.0, 400.0, 11.0):
        ref = sp.spkpos("MOON", et(JD0 + d), "J2000", "NONE", "EARTH")[0]
        assert np.linalg.norm(moon_geocentric(JD0 + d) - ref) < 800.0     # km
