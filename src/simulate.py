"""Quasi-static time-series loop for the CIGRE MV feeder with PV.

Every controller study of this series runs through the same loop: the
circuit is built once (radial feeder 1, the nine PV units of cigre_dss.PV_KW
with an inverter rating of KVA_RATIO x Pmpp, TR1 tap fixed at TR1_TAP_PCT),
and for each step of the stored year (profiles_2023.h5) the two load sectors
are scaled, the nine irradiances are set, an optional per-step controller is
called, the snapshot is solved and the feeder-1 voltages, PV P/Q and total
losses are recorded.

Study configuration: TR1 at +4.375 % on the 20 kV side (the tap at which the
feeder without PV stays inside 0.95 to 1.05 pu at every load level, so that
every violation is attributable to PV) and kVA = 1.1 x Pmpp (the oversizing
that gives the IEEE 1547-2018 Category B reactive capability of 0.44 pu of
kVA at full active output).

Controllers come in two forms:
  * an OpenDSS InvControl (volt-var, volt-watt): `controller_setup_fn` adds
    the XYcurve/InvControl elements after the circuit is built and returns
    None; OpenDSS then iterates the control loop inside every solve
    (controlmode=static);
  * a Python per-step controller (the OPF bound, or a learned policy):
    `controller_setup_fn` returns a callable step_fn(i, irr, res, com) that
    is invoked before the solve of step i and sets the kvar setpoints of the
    PV units (see set_pv_q).

Everything is returned as numpy arrays inside a YearResult so that kpi.py
can score any controller with the same code.
"""
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import opendssdirect as dss
from dss import DSSException

import cigre_dss as cd
import profiles as pf

TR1_TAP_PCT = 4.375              # % on the 20 kV side (brochure value: 6.25)
KVA_RATIO = 1.1                  # inverter kVA / Pmpp
PV_BUSES = list(cd.PV_KW)
PV_KW = np.array([cd.PV_KW[b] for b in PV_BUSES], dtype=float)
PV_KVA = KVA_RATIO * PV_KW
F1_BUSES = list(range(1, 12))    # feeder 1 (TR1 secondary to the far end)
DT_H = 0.25
V_HI, V_LO = 1.05, 0.95
CUTIN_PU = 0.001                 # %cutin = %cutout = 0.1 (of kVA) in cigre_dss.add_pv

_node_idx = None                 # per feeder-1 bus: indices of its 3 nodes in AllBusMagPu


def load_year(path="profiles_2023.h5"):
    """(pv DataFrame T x 9, loads DataFrame, attrs) of the stored year."""
    return pf.load_profiles_from(path)


def build(tr1_tap_pct=TR1_TAP_PCT, closed_switches=(), kva_ratio=KVA_RATIO):
    """Fresh circuit with the nine PV units (unity pf, kva = kva_ratio x Pmpp)
    and static control mode; no control element yet."""
    global _node_idx
    cd.build_circuit(closed_switches=closed_switches, tr1_tap_pct=tr1_tap_pct)
    cd.add_pv(cd.PV_KW, kva_ratio=kva_ratio)
    dss.Text.Command("set controlmode=static")
    dss.Text.Command("set maxcontroliter=200")
    names = dss.Circuit.AllBusNames()
    _node_idx = np.array([[3 * names.index(str(b)) + k for k in range(3)] for b in F1_BUSES])


def solve_step(iters=3):
    """Solve the snapshot, re-holding bus 0 at 110 kV (as cd.solve, but with
    fewer source iterations because the source setting of the previous step
    is a good start). Returns True if the OpenDSS control loop hit
    maxcontroliter at least once; the last solution is kept in that case."""
    failed = False
    for k in range(iters + 1):
        try:
            dss.Solution.Solve()
        except DSSException as e:
            if "Max Control Iterations" not in str(e):
                raise
            failed = True
        if k < iters:
            dss.Circuit.SetActiveBus("0")
            va = dss.Bus.VMagAngle()
            v0 = (va[0] + va[2] + va[4]) / 3 * np.sqrt(3) / 1e3
            dss.Vsources.First()
            dss.Vsources.PU(dss.Vsources.PU() * 110.0 / v0)
            dss.Vsources.AngleDeg(dss.Vsources.AngleDeg() - va[1])
    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS did not converge")
    return failed


def feeder1_voltages():
    """Voltages of buses 1..11 in pu (mean of the three phases)."""
    v = np.asarray(dss.Circuit.AllBusMagPu())
    return v[_node_idx].mean(axis=1)


def pv_pq():
    """(P kW, Q kvar) injected by the nine units, in PV_BUSES order."""
    p = np.empty(len(PV_BUSES)); q = np.empty(len(PV_BUSES))
    for j, b in enumerate(PV_BUSES):
        dss.Circuit.SetActiveElement(f"pvsystem.pv{b}")
        s = dss.CktElement.Powers()
        p[j] = -(s[0] + s[2] + s[4]); q[j] = -(s[1] + s[3] + s[5])
    return p, q


def set_pv_q(q_kvar):
    """Fixed kvar setpoint of each unit (positive = injecting), PV_BUSES order."""
    for j, b in enumerate(PV_BUSES):
        dss.PVsystems.Name(f"pv{b}")
        dss.PVsystems.kvar(float(q_kvar[j]))


def losses_kw():
    return dss.Circuit.Losses()[0] / 1e3


def add_voltvar(x, y, name="vv", der_list=None):
    """IEEE 1547 style volt-var InvControl on every PV unit. x: voltages in pu
    of rated; y: Q in pu of kVA (refreactivepower=varmax makes the y axis a
    fraction of the inverter kVA rather than of the vars still available at
    the current P)."""
    dss.Text.Command(f"new xycurve.{name} npts={len(x)} xarray=({','.join(map(str, x))}) "
                     f"yarray=({','.join(map(str, y))})")
    dss.Text.Command(f"new invcontrol.ic_{name} mode=voltvar voltage_curvex_ref=rated "
                     f"vvc_curve1={name} deltaq_factor=-1 refreactivepower=varmax")


def add_voltvar_voltwatt(x_vv, y_vv, x_vw, y_vw, name="vvvw"):
    """Combined volt-var + volt-watt (InvControl combimode=VV_VW). The
    volt-watt y axis is P in pu of Pmpp."""
    dss.Text.Command(f"new xycurve.{name}_vv npts={len(x_vv)} xarray=({','.join(map(str, x_vv))}) "
                     f"yarray=({','.join(map(str, y_vv))})")
    dss.Text.Command(f"new xycurve.{name}_vw npts={len(x_vw)} xarray=({','.join(map(str, x_vw))}) "
                     f"yarray=({','.join(map(str, y_vw))})")
    dss.Text.Command(f"new invcontrol.ic_{name} combimode=VV_VW voltage_curvex_ref=rated "
                     f"vvc_curve1={name}_vv voltwatt_curve={name}_vw voltwattyaxis=PMPPPU "
                     f"deltaq_factor=-1 deltap_factor=-1 refreactivepower=varmax")


@dataclass
class YearResult:
    label: str
    index: pd.DatetimeIndex
    steps: np.ndarray            # indices into the year of the simulated steps
    v: np.ndarray                # (n, 11) pu, buses 1..11
    p_kw: np.ndarray             # (n, 9) PV active power
    q_kvar: np.ndarray           # (n, 9) PV reactive power (positive = injecting)
    irr: np.ndarray              # (n, 9) irradiance pu
    res: np.ndarray              # (n,) residential load pu
    com: np.ndarray              # (n,) commercial/industrial load pu
    losses_kw: np.ndarray        # (n,) total circuit losses
    failed: np.ndarray           # (n,) bool, control loop hit maxcontroliter
    runtime_s: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def p_avail_kw(self):
        """Available PV power Pmpp * irradiance, with the inverter cut-in
        applied (cigre_dss.add_pv sets %cutin = %cutout = 0.1, i.e. the unit
        is off below 0.1 % of kVA), so that a unit that is off at dawn or dusk
        is not counted as curtailed."""
        p = self.irr * PV_KW
        return np.where(p >= CUTIN_PU * PV_KVA, p, 0.0)

    def day_mask(self, day):
        d = pd.Timestamp(day)
        return (self.index >= d) & (self.index < d + pd.Timedelta(days=1))

    def subset(self, mask):
        return YearResult(self.label, self.index[mask], self.steps[mask], self.v[mask], self.p_kw[mask],
                          self.q_kvar[mask], self.irr[mask], self.res[mask], self.com[mask],
                          self.losses_kw[mask], self.failed[mask], self.runtime_s, dict(self.extra))


def run_year(controller_setup_fn=None, steps=None, pv=None, loads=None, label="",
             tr1_tap_pct=TR1_TAP_PCT, verbose=True, kva_ratio=KVA_RATIO):
    """Run the stored year (or the subset `steps`, an array of step indices)
    with the given controller. See the module docstring for the two
    controller forms. Returns a YearResult."""
    if pv is None or loads is None:
        pv, loads, _ = load_year()
    irr_all = pv.values.astype(float)
    res_all = loads["residential"].values.astype(float)
    com_all = loads["commercial_industrial"].values.astype(float)
    steps = np.arange(len(pv)) if steps is None else np.asarray(steps, dtype=int)
    n = len(steps)

    build(tr1_tap_pct=tr1_tap_pct, kva_ratio=kva_ratio)
    step_fn = controller_setup_fn() if controller_setup_fn is not None else None
    if step_fn is not None and not callable(step_fn):
        step_fn = None

    v = np.empty((n, len(F1_BUSES))); p = np.empty((n, len(PV_BUSES))); q = np.empty((n, len(PV_BUSES)))
    loss = np.empty(n); failed = np.zeros(n, dtype=bool)
    t0 = time.perf_counter()
    for k, i in enumerate(steps):
        cd.scale_loads(res_all[i], com_all[i])
        for j, b in enumerate(PV_BUSES):
            dss.PVsystems.Name(f"pv{b}")
            dss.PVsystems.Irradiance(float(irr_all[i, j]))
        if step_fn is not None:
            step_fn(i, irr_all[i], res_all[i], com_all[i])
        failed[k] = solve_step()
        v[k] = feeder1_voltages()
        p[k], q[k] = pv_pq()
        loss[k] = losses_kw()
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"{label or 'run':28s}: {n} steps in {elapsed:6.1f} s ({elapsed / n * 1e3:.2f} ms per step), "
              f"control-loop failures {int(failed.sum())}")
    return YearResult(label, pv.index[steps], steps, v, p, q, irr_all[steps], res_all[steps], com_all[steps],
                      loss, failed, elapsed)


# --------------------------------------------------------------------------
# Per-step OPF (the feeder-wide bound)
# --------------------------------------------------------------------------

class OPFController:
    """Per-step dispatch of the nine PV units by direct optimisation, with
    OpenDSS solving the power flow inside the objective.

    Two objectives are available.

    objective="losses" (default; after Wagle et al. 2023, IET GTD, Eq. 2):

        J(q) = P_loss(q) [kW] + w_viol * N_viol(q) + w_q * sum |q_j| [kvar]

    over q_j in [-q_max_pu, +q_max_pu] * kVA_j, where N_viol is the number of
    feeder-1 buses outside [v_lo, v_hi].

    objective="reward": minus the per-step reward of the learning environment
    of the later notebooks,

        J(q, c) = w_viol * sum_b max(0, V_b - v_hi, v_lo - V_b) [pu]
                  + w_loss * P_loss [kW] + w_q * sum |q_j| [kvar] + w_curt * P_curt [kW]

    with optional per-unit curtailment fractions c_j in [0, 1] (curtail=True)
    that scale the irradiance of unit j, P_curt = sum c_j * irr_j * Pmpp_j, and
    the inverter capability q_j^2 + P_j^2 <= kVA_j^2 as a constraint
    (capability=True), so that the bound is comparable with controllers whose
    reactive range is limited by the same capability.

    Both objectives contain a step or hinge, so the controller solves, from a
    warm start (previous step's x), the equivalent smooth problem

        min  smooth part of J   s.t.  v_lo <= V_b(x) <= v_hi  (and capability)

    with scipy's SLSQP (finite-difference gradients, variables in pu of kVA
    and curtailment fraction); if SLSQP does not report success or the result
    still violates a limit, a penalty form with a hinge term w_hinge * sum
    max(0, V-v_hi, v_lo-V) is solved as a fallback. The candidates (warm
    start, x = 0, the SLSQP results) are then scored with the stated J and the
    best one is applied, so the applied setpoint is never worse than doing
    nothing or keeping the previous setpoint under J itself. `log` keeps J,
    the number of power flows and the time of every step."""

    def __init__(self, w_q=0.01, w_viol=1000.0, q_max_pu=0.44, w_hinge=1e5,
                 v_hi=V_HI, v_lo=V_LO, maxiter=100, ftol=1e-4, eps=1e-3,
                 objective="losses", w_loss=1.0, w_curt=0.05, curtail=False, capability=False,
                 pv_kva=None, pv_kw=None):
        self.w_q, self.w_viol, self.q_max_pu, self.w_hinge = w_q, w_viol, q_max_pu, w_hinge
        self.v_hi, self.v_lo = v_hi, v_lo
        if objective == "nb04":                  # alias of the loss objective of notebook 04
            objective = "losses"
        self.objective_kind, self.w_loss, self.w_curt = objective, w_loss, w_curt
        self.curtail, self.capability = curtail, capability
        self.pv_kva = PV_KVA if pv_kva is None else np.asarray(pv_kva, dtype=float)
        self.pv_kw = PV_KW if pv_kw is None else np.asarray(pv_kw, dtype=float)
        self.opts = dict(maxiter=maxiter, ftol=ftol, eps=eps)
        self.n = len(PV_BUSES)
        self.nx = 2 * self.n if curtail else self.n
        self.x_prev = np.zeros(self.nx)
        self.log = []
        self._cache = {}
        self._irr = np.zeros(self.n)

    # run_year calls this as controller_setup_fn: nothing to add to the
    # circuit, the per-step function is returned instead
    def __call__(self):
        self.x_prev[:] = 0.0
        self.log.clear()
        return self.step

    def _split(self, x):
        q = x[:self.n]
        c = x[self.n:] if self.curtail else np.zeros(self.n)
        return q, c

    def _apply(self, x):
        q, c = self._split(x)
        set_pv_q(q * self.pv_kva)
        if self.curtail:
            for j, b in enumerate(PV_BUSES):
                dss.PVsystems.Name(f"pv{b}")
                dss.PVsystems.Irradiance(float(self._irr[j] * (1.0 - c[j])))

    def _evaluate(self, x):
        key = x.tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        self._apply(x)
        solve_step(iters=1)
        p, q = pv_pq()
        out = (losses_kw(), feeder1_voltages(), p, q)
        self._cache[key] = out
        return out

    def _curt_kw(self, x, p=None):
        """Curtailed active power, kW: what the profile would give minus what
        the units deliver after the explicit curtailment c and after any
        implicit clipping by OpenDSS at the kVA limit."""
        if p is None:
            p = self._evaluate(x)[2]
        return float(np.maximum(0.0, self._irr * self.pv_kw - p).sum())

    def objective(self, x):
        """The stated J at setpoint x (pu of kVA, and curtailment fraction)."""
        L, v, p, q = self._evaluate(x)
        qk = self._split(x)[0] * self.pv_kva
        if self.objective_kind == "losses":
            n_viol = int(((v > self.v_hi) | (v < self.v_lo)).sum())
            return L + self.w_viol * n_viol + self.w_q * np.abs(qk).sum()
        excess = np.maximum(0.0, np.maximum(v - self.v_hi, self.v_lo - v)).sum()
        return (self.w_viol * excess + self.w_loss * L + self.w_q * np.abs(q).sum()
                + self.w_curt * self._curt_kw(x, p))

    def _smooth_loss(self, x):
        L, v, p, q = self._evaluate(x)
        qk = self._split(x)[0] * self.pv_kva
        if self.objective_kind == "losses":
            return L + self.w_q * np.sqrt(qk ** 2 + 1.0).sum()
        return self.w_loss * L + self.w_q * np.sqrt(qk ** 2 + 1.0).sum() + self.w_curt * self._curt_kw(x, p)

    def _constraints(self, x):
        _, v, p, q = self._evaluate(x)
        cons = [self.v_hi - 1e-5 - v, v - self.v_lo - 1e-5]
        if self.capability:                      # on the available P, before any clipping by OpenDSS
            qq, c = self._split(x)
            p_av = self._irr * (1.0 - c) * self.pv_kw
            cons.append((self.pv_kva ** 2 - p_av ** 2 - (qq * self.pv_kva) ** 2) / self.pv_kva ** 2)
        return np.concatenate(cons)

    def _penalty(self, x):
        L, v, p, q = self._evaluate(x)
        hinge = np.maximum(0.0, np.maximum(v - self.v_hi, self.v_lo - v)).sum()
        if self.capability:
            qq, c = self._split(x)
            p_av = self._irr * (1.0 - c) * self.pv_kw
            hinge += np.maximum(0.0, (p_av ** 2 + (qq * self.pv_kva) ** 2 - self.pv_kva ** 2) / self.pv_kva ** 2).sum()
        return self._smooth_loss(x) + self.w_hinge * hinge

    def step(self, i, irr, res, com):
        from scipy.optimize import minimize
        t0 = time.perf_counter()
        self._cache.clear()
        self._irr = np.asarray(irr, dtype=float)
        bounds = [(-self.q_max_pu, self.q_max_pu)] * self.n
        if self.curtail:
            day = self._irr > 0
            bounds += [(0.0, 1.0 if day[j] else 0.0) for j in range(self.n)]     # no curtailment variable at night
        x0 = self.x_prev.copy()
        if self.curtail:
            x0[self.n:] = np.where(self._irr > 0, x0[self.n:], 0.0)
        cands = [np.zeros(self.nx), x0]
        method = "constrained"
        r = minimize(self._smooth_loss, x0, method="SLSQP", bounds=bounds,
                     constraints=[dict(type="ineq", fun=self._constraints)], options=self.opts)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        cands.append(np.clip(r.x, lo, hi))
        _, v, _, _ = self._evaluate(cands[-1])
        if (not r.success) or (v > self.v_hi).any() or (v < self.v_lo).any():
            method = "penalty"
            best0 = min(cands, key=self.objective)
            r2 = minimize(self._penalty, best0, method="SLSQP", bounds=bounds, options=self.opts)
            cands.append(np.clip(r2.x, lo, hi))
        scores = [self.objective(x) for x in cands]
        k = int(np.argmin(scores))
        x = cands[k]
        self._apply(x)
        self.x_prev = x.copy()
        self.log.append(dict(step=int(i), J=float(scores[k]), n_pf=len(self._cache),
                             time_s=time.perf_counter() - t0, method=method,
                             chosen=("zero", "warm", "slsqp", "penalty")[k],
                             curtailed_kw=self._curt_kw(x)))
