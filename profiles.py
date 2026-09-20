"""Synthetic one-year PV and load profiles for the CIGRE European MV feeder.

Everything here is generated from closed-form solar geometry, the brochure's
daily load shapes (CIGRE TB 575, Figure 6.4, p.51) and a seeded random number
generator; nothing is downloaded. Time stamps are local *solar* time of the
site (no time zone, no daylight saving, no equation of time), so 12:00 is
solar noon on every day.

PV model
--------
Clear-sky irradiance on the array plane, from plain solar geometry:

  declination        delta = 23.45 deg * sin(2 pi (284 + n) / 365)      [Cooper 1969]
  hour angle         omega = 15 deg/h * (t_solar - 12 h)
  solar elevation    sin(el) = sin(phi) sin(delta) + cos(phi) cos(delta) cos(omega)
  clear-sky GHI      G_clear = 1098 W/m2 * sin(el) * exp(-0.059 / sin(el))   [Haurwitz 1945]

The exp(-0.059/sin el) term is the air-mass attenuation of the Haurwitz
model (air mass ~ 1/sin el). Plane-of-array simplification: the array is
south facing at a fixed tilt beta, and only the beam component is projected,
so the incidence angle theta is the solar zenith angle seen from an
"effective latitude" phi - beta:

  cos(theta) = sin(phi - beta) sin(delta) + cos(phi - beta) cos(delta) cos(omega)
  G_poa      = 1098 * max(cos theta, 0) * exp(-0.059 / sin(el))          for el > 0

The per-unit clear-sky output is G_poa / 1000 W/m2, capped at 1 (Pmpp is
quoted at 1 kW/m2, 25 C; no temperature derating, as in cigre_dss.add_pv).

Cloud model: a two-state Markov chain (clear / cloudy) with a seasonal
stationary cloudy fraction

  pi_cloudy(n) = 0.60 + 0.20 cos(2 pi (n - 15) / 365)      (winter ~0.80, summer ~0.40)

and a mean clear-spell length of 3 h. In the clear state the clearness index
is k = 0.92 - 0.03 |N(0,1)| (the Haurwitz model overestimates measured
clear-sky irradiance by a few per cent, and some haze is always present); in
the cloudy state k follows a Beta(2, 4) draw per step (mean 0.33) passed
through an AR(1) smoother (rho = 0.5), except that with probability 0.15 per
step the smoother is bypassed, which gives the occasional fast ramp of a
passing cloud edge. Output pu = clear-sky pu * k.

Spatial diversity: the nine units share the same k, but each unit gets an
independent multiplicative factor 1 + 0.05 u_i(t), u_i an AR(1) process with
unit variance clipped to [-1, 1], i.e. +/-5 % at most.

Load model
----------
Brochure daily shapes (30 min, pu of node peak apparent power) interpolated
linearly to the step, times a seasonal factor

  residential            1 + 0.15 cos(2 pi (n - 15) / 365)
  commercial/industrial  1 + 0.05 cos(2 pi (n - 15) / 365)

times a weekday/weekend factor (Sat/Sun: residential 1.05, commercial 0.60),
times (1 + e), e Gaussian noise smoothed with a 1 h moving average and scaled
to sigma = 3 %. Clipped to >= 0.05 and normalised so the annual maximum of
each sector is 1.0.
"""
import json
import math
import os

import h5py
import numpy as np
import pandas as pd

# Same data file as cigre_dss.py (data/ folder next to this module).
DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                            "cigre_mv_european_tb575.json")

# clear-sky / cloud parameters (documented in the module docstring)
HAURWITZ_A, HAURWITZ_B = 1098.0, 0.059     # W/m2, air-mass exponent
TILT_DEG = 30.0                            # south-facing array tilt
CLOUD_MEAN, CLOUD_AMP, CLOUD_PHASE = 0.60, 0.20, 15   # pi_cloudy(n)
CLEAR_SPELL_H = 3.0                        # mean length of a clear spell
BETA_A, BETA_B = 2.0, 4.0                  # cloudy-state clearness
AR_RHO, RAMP_P = 0.5, 0.15                 # smoother and fast-ramp probability
CLEAR_K, HAZE_SIGMA = 0.92, 0.03           # clear-state clearness
DIVERSITY = 0.05                           # +/- 5 % between units
DIVERSITY_RHO = 0.8                        # AR(1) coefficient of the unit factors


def time_index(year=2023, step_min=15):
    """Solar-time index of the whole year at the given step (no time zone)."""
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    return pd.date_range(start, end, freq=f"{step_min}min", inclusive="left")


# --------------------------------------------------------------------------
# PV
# --------------------------------------------------------------------------

def clear_sky_pu(idx, lat_deg=50.0, tilt_deg=TILT_DEG):
    """Clear-sky per-unit output (0..1 of Pmpp) at each time stamp of `idx`."""
    n = idx.dayofyear.values                                  # 1..365
    t = idx.hour.values + idx.minute.values / 60.0            # solar hours
    delta = math.radians(23.45) * np.sin(2 * np.pi * (284 + n) / 365)      # declination
    omega = np.radians(15.0 * (t - 12.0))                                   # hour angle
    phi = math.radians(lat_deg)
    sin_el = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(omega)
    phi_eff = math.radians(lat_deg - tilt_deg)                             # tilted plane
    cos_th = np.sin(phi_eff) * np.sin(delta) + np.cos(phi_eff) * np.cos(delta) * np.cos(omega)
    up = sin_el > 0.0
    g = np.zeros(len(idx))
    g[up] = HAURWITZ_A * np.maximum(cos_th[up], 0.0) * np.exp(-HAURWITZ_B / sin_el[up])
    return np.clip(g / 1000.0, 0.0, 1.0)


def _clearness(idx, rng):
    """Clearness index k(t) in [0, 1] from the two-state Markov cloud model."""
    T = len(idx)
    n = idx.dayofyear.values
    step_h = (idx[1] - idx[0]).total_seconds() / 3600.0
    pi_c = CLOUD_MEAN + CLOUD_AMP * np.cos(2 * np.pi * (n - CLOUD_PHASE) / 365)
    a = 1.0 - step_h / CLEAR_SPELL_H                        # P(clear -> clear)
    # stationary cloudy fraction pi_c = (1-a) / ((1-a) + (1-b))  ->  b
    b = 1.0 - (1.0 - a) * (1.0 - pi_c) / pi_c                # P(cloudy -> cloudy)
    u = rng.random(T)
    beta = rng.beta(BETA_A, BETA_B, T)
    haze = CLEAR_K - HAZE_SIGMA * np.abs(rng.standard_normal(T))
    ramp = rng.random(T) < RAMP_P
    cloudy = np.zeros(T, dtype=bool)
    k = np.ones(T)
    state = u[0] < pi_c[0]
    k_prev = beta[0] if state else haze[0]
    for i in range(T):
        if i > 0:
            stay = b[i] if state else a
            if u[i] >= stay:
                state = not state
        cloudy[i] = state
        if state:
            k_i = beta[i] if (ramp[i] or not cloudy[i - 1]) else AR_RHO * k_prev + (1 - AR_RHO) * beta[i]
        else:
            k_i = haze[i]
        k[i] = k_i
        k_prev = k_i
    return np.clip(k, 0.0, 1.0), cloudy


def pv_profile(year=2023, lat_deg=50.0, step_min=15, seed=0, return_parts=False):
    """Per-unit PV output (0..1 of Pmpp) of a single array for the whole year.

    return_parts=True also returns a DataFrame with the clear-sky pu, the
    clearness index and the cloudy-state flag."""
    idx = time_index(year, step_min)
    rng = np.random.default_rng(seed)
    cs = clear_sky_pu(idx, lat_deg)
    k, cloudy = _clearness(idx, rng)
    pv = pd.Series(np.clip(cs * k, 0.0, 1.0), index=idx, name="pv_pu")
    if return_parts:
        parts = pd.DataFrame({"clear_sky": cs, "clearness": k, "cloudy": cloudy}, index=idx)
        return pv, parts
    return pv


def _ar1(rng, T, rho):
    """Unit-variance AR(1) process of length T."""
    e = rng.standard_normal(T) * math.sqrt(1 - rho ** 2)
    x = np.empty(T)
    x[0] = rng.standard_normal()
    for i in range(1, T):
        x[i] = rho * x[i - 1] + e[i]
    return x


def pv_profiles(n_units=9, names=None, year=2023, lat_deg=50.0, step_min=15, seed=0):
    """DataFrame (T x n_units) of per-unit PV output, one column per unit.
    All units see the same irradiance; each has an independent +/-5 % factor."""
    base = pv_profile(year, lat_deg, step_min, seed)
    rng = np.random.default_rng(seed + 1000)
    T = len(base)
    cols = {}
    names = list(names) if names is not None else list(range(n_units))
    for name in names:
        u = np.clip(_ar1(rng, T, DIVERSITY_RHO), -1.0, 1.0)
        cols[name] = np.clip(base.values * (1.0 + DIVERSITY * u), 0.0, 1.0)
    return pd.DataFrame(cols, index=base.index)


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

SEASON = {"residential": 0.15, "commercial_industrial": 0.05}
SEASON_PHASE = 15                          # day of year of the seasonal maximum
WEEKEND = {"residential": 1.05, "commercial_industrial": 0.60}
NOISE_SIGMA = 0.03


def daily_shapes(data_path=DEFAULT_DATA, step_min=15):
    """Brochure daily shapes (Figure 6.4) interpolated to step_min; dict
    sector -> array of 24 h / step_min values starting at 00:00."""
    with open(data_path, encoding="utf-8") as f:
        prof = json.load(f)["load_profiles"]
    t_out = np.arange(0, 24, step_min / 60.0)
    out = {}
    for sec in ("residential", "commercial_industrial"):
        t = np.array([r["t_h"] for r in prof[sec]])
        s = np.array([r["S_pu_of_max"] for r in prof[sec]])
        out[sec] = np.interp(t_out, t, s)
    return out


def seasonal_factor(n, sector):
    """Seasonal multiplier of a sector at day of year n (array or scalar)."""
    return 1 + SEASON[sector] * np.cos(2 * np.pi * (np.asarray(n) - SEASON_PHASE) / 365)


def load_profiles(year=2023, step_min=15, seed=0, data_path=DEFAULT_DATA):
    """DataFrame with columns residential, commercial_industrial: load in pu
    of annual peak apparent power for the whole year."""
    idx = time_index(year, step_min)
    rng = np.random.default_rng(seed + 2000)
    shapes = daily_shapes(data_path, step_min)
    T = len(idx)
    n = idx.dayofyear.values
    weekend = idx.dayofweek.values >= 5
    slot = (idx.hour.values * 60 + idx.minute.values) // step_min
    win = max(int(60 / step_min), 1)
    out = {}
    for sec in ("residential", "commercial_industrial"):
        x = shapes[sec][slot]
        x = x * seasonal_factor(n, sec)
        x = x * np.where(weekend, WEEKEND[sec], 1.0)
        e = rng.standard_normal(T + win - 1)
        e = np.convolve(e, np.ones(win) / win, mode="valid")     # 1 h moving average
        e = e / e.std() * NOISE_SIGMA
        x = x * (1 + e)
        x = np.maximum(x, 0.05)
        out[sec] = x / x.max()
    return pd.DataFrame(out, index=idx)


# --------------------------------------------------------------------------
# File I/O (h5py; pandas HDFStore needs pytables, which is not installed)
# --------------------------------------------------------------------------

def save_profiles(path, pv, loads, attrs=None):
    """Write pv (DataFrame T x n) and loads (DataFrame with residential and
    commercial_industrial) to one HDF5 file with datasets time (epoch
    seconds, int64), pv, res, com."""
    t = (pv.index.asi8 // 10 ** 9).astype(np.int64)
    with h5py.File(path, "w") as f:
        f.create_dataset("time", data=t)
        f["time"].attrs["units"] = "seconds since 1970-01-01 00:00, local solar time"
        f.create_dataset("pv", data=pv.values.astype(np.float32), compression="gzip")
        f["pv"].attrs["columns"] = [str(c) for c in pv.columns]
        f["pv"].attrs["units"] = "pu of Pmpp"
        f.create_dataset("res", data=loads["residential"].values.astype(np.float32), compression="gzip")
        f.create_dataset("com", data=loads["commercial_industrial"].values.astype(np.float32), compression="gzip")
        f["res"].attrs["units"] = f["com"].attrs["units"] = "pu of annual peak apparent power"
        for k, v in (attrs or {}).items():
            f.attrs[k] = v


def load_profiles_from(path):
    """Read a file written by save_profiles. Returns (pv DataFrame, loads
    DataFrame, dict of file attributes)."""
    with h5py.File(path, "r") as f:
        idx = pd.to_datetime(f["time"][:].astype(np.int64), unit="s")
        cols = [c for c in f["pv"].attrs["columns"]]
        cols = [int(c) if c.lstrip("-").isdigit() else c for c in cols]
        pv = pd.DataFrame(f["pv"][:].astype(float), index=idx, columns=cols)
        loads = pd.DataFrame({"residential": f["res"][:].astype(float),
                              "commercial_industrial": f["com"][:].astype(float)}, index=idx)
        attrs = {k: (v.tolist() if isinstance(v, np.ndarray) else (v.item() if hasattr(v, "item") else v))
                 for k, v in f.attrs.items()}
    return pv, loads, attrs
