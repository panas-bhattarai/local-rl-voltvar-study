"""Key performance indicators of a voltage controller on the CIGRE MV feeder.

Every controller of this series is scored with compute_kpis on a
simulate.YearResult (or on a subset of one), so that the numbers are
comparable between notebooks. Definitions (dt = step length in hours,
buses 1..11 of feeder 1, limits 0.95 / 1.05 pu):

  hours_violation   hours with at least one bus outside the band
  hours_over        hours with at least one bus above 1.05 pu
  hours_under       hours with at least one bus below 0.95 pu
  vpi               voltage performance index, pu.h:
                    sum over steps and buses of max(0, V-1.05, 0.95-V) * dt
  losses_mwh        total circuit losses (dss.Circuit.Losses), MWh
  curtailed_mwh     PV energy not delivered, MWh: sum(Pmpp*irr - P) * dt
  reactive_mvarh    reactive energy exchanged by the PV units, Mvarh: sum|Q| * dt
  reactive_absorbed_mvarh, reactive_injected_mvarh
                    the two signs of the same sum (absorbed = Q < 0)
  pv_energy_mwh     PV energy delivered, MWh
  n_fail            steps at which the OpenDSS control loop hit maxcontroliter
  vmax, vmin        extreme feeder-1 voltages over the period
  hours             length of the scored period
"""
import numpy as np
import pandas as pd

KPI_COLUMNS = ["hours_violation", "hours_over", "hours_under", "vpi", "losses_mwh",
               "curtailed_mwh", "reactive_mvarh", "reactive_absorbed_mvarh", "reactive_injected_mvarh",
               "pv_energy_mwh", "n_fail", "vmax", "vmin", "hours"]

KPI_LABELS = {"hours_violation": "hours outside 0.95-1.05 pu", "hours_over": "hours above 1.05 pu",
              "hours_under": "hours below 0.95 pu", "vpi": "VPI, pu.h", "losses_mwh": "losses, MWh",
              "curtailed_mwh": "curtailed PV, MWh", "reactive_mvarh": "reactive energy, Mvarh",
              "reactive_absorbed_mvarh": "reactive energy absorbed, Mvarh",
              "reactive_injected_mvarh": "reactive energy injected, Mvarh",
              "pv_energy_mwh": "PV energy, MWh", "n_fail": "control-loop failures",
              "vmax": "max voltage, pu", "vmin": "min voltage, pu", "hours": "period, h"}


def compute_kpis(v, p_kw, q_kvar, p_avail_kw, losses_kw, failed=None, dt_h=0.25,
                 v_hi=1.05, v_lo=0.95):
    """KPIs from per-step arrays: v (n, nbus), p_kw / q_kvar / p_avail_kw
    (n, nunits), losses_kw (n,), failed (n,) bool. Returns a dict."""
    v = np.asarray(v); over = v > v_hi; under = v < v_lo
    excess = np.maximum(0.0, np.maximum(v - v_hi, v_lo - v))
    curt = np.maximum(0.0, np.asarray(p_avail_kw) - np.asarray(p_kw))
    q = np.asarray(q_kvar)
    return dict(
        hours_violation=float((over | under).any(axis=1).sum() * dt_h),
        hours_over=float(over.any(axis=1).sum() * dt_h),
        hours_under=float(under.any(axis=1).sum() * dt_h),
        vpi=float(excess.sum() * dt_h),
        losses_mwh=float(np.sum(losses_kw) * dt_h / 1e3),
        curtailed_mwh=float(curt.sum() * dt_h / 1e3),
        reactive_mvarh=float(np.abs(q).sum() * dt_h / 1e3),
        reactive_absorbed_mvarh=float(-q[q < 0].sum() * dt_h / 1e3),
        reactive_injected_mvarh=float(q[q > 0].sum() * dt_h / 1e3),
        pv_energy_mwh=float(np.sum(p_kw) * dt_h / 1e3),
        n_fail=int(np.sum(failed)) if failed is not None else 0,
        vmax=float(v.max()), vmin=float(v.min()), hours=float(len(v) * dt_h))


def kpis_of(result, mask=None, dt_h=0.25):
    """compute_kpis on a simulate.YearResult (optionally on the steps where
    mask is True)."""
    r = result if mask is None else result.subset(mask)
    return compute_kpis(r.v, r.p_kw, r.q_kvar, r.p_avail_kw, r.losses_kw, r.failed, dt_h)


def kpi_table(kpis_by_name, columns=None):
    """DataFrame, one row per controller."""
    cols = columns or KPI_COLUMNS
    return pd.DataFrame({n: [k[c] for c in cols] for n, k in kpis_by_name.items()}, index=cols).T


def format_table(df):
    """Fixed-format string of a kpi_table for printing."""
    fmt = {"hours_violation": "{:9.2f}", "hours_over": "{:9.2f}", "hours_under": "{:9.2f}", "vpi": "{:9.4f}",
           "losses_mwh": "{:9.2f}", "curtailed_mwh": "{:9.3f}", "reactive_mvarh": "{:9.2f}",
           "reactive_absorbed_mvarh": "{:9.2f}", "reactive_injected_mvarh": "{:9.2f}",
           "pv_energy_mwh": "{:9.2f}", "n_fail": "{:9.0f}", "vmax": "{:9.4f}", "vmin": "{:9.4f}", "hours": "{:9.0f}"}
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(lambda x, f=fmt.get(c, "{:9.3f}"): f.format(x))
    return out.to_string()
