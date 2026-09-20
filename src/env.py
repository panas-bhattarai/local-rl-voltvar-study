"""Learning environment for volt-var curve tuning on the CIGRE MV feeder.

One environment step is one 15-minute step of the stored year
(profiles_2023.h5). At every step each of the nine PV units receives its own
volt-var curve from an action vector, OpenDSS solves the snapshot with one
InvControl per unit (controlmode=static, as in simulate.run_year), a safety
layer corrects the result if a bus of feeder 1 is outside 0.95 to 1.05 pu,
and the step returns per-unit observations, one shared reward and a dict of
diagnostics. The inverter rating is kva_ratio x Pmpp (EnvConfig.kva_ratio,
1.1 as in notebooks 02 to 04); the ratio enters the PVSystem kVA, the
observation scaling and the capability bound of the safety layer.

Curve parameterisation (IEEE 1547-2018 Table 8, Category B, VRef = 1.0 pu):

    x = (V1, V2, V3, V4) = (0.92, 0.98, V3, V4)      pu of rated voltage
    y = (Q1, Q2, Q3, Q4) = (+0.44, 0, 0, -Q4)         pu of inverter kVA

V1, Q1 and V2 are fixed (undervoltage does not occur on this feeder,
notebook 04). The action a in [-1, 1]^3 is mapped affinely to

    V3 in [1.00, 1.03],  V4 in [V3 + 0.02, 1.18],  Q4 in [0, 0.44]

so that V2 < V3 < V4 holds by construction. DEFAULT_ACTION reproduces the
Category B default (V3 = 1.02, V4 = 1.08, Q4 = 0.44) of controller B1,
TIGHT_ACTION the curve of B3 and LAZY_ACTION the curve of B4.

The curves are edited in place through the XYCurves API of opendssdirect
(XArray / YArray); OpenDSS's InvControl reads the curve at every control
iteration, so no element is recreated between steps.

OpenDSS holds a single circuit per process, so only one environment is live
at a time: reset() rebuilds the circuit (a few milliseconds), and an episode
of one environment must not be interleaved with steps of another.

Reward (shared by all units, per step):

    r = - w_viol * sum_b max(0, V_b - 1.05, 0.95 - V_b)     [pu, buses 1..11]
        - w_loss * losses_kW
        - w_q    * sum_i |Q_i|                               [kvar]
        - w_curt * curtailed_kW
        - w_safety * (fence_dQ_kvar + 10 * fence_dP_kW)      [fence charge]
        - w_safety_event * [fence acted on this step]

The first four terms are the core reward (info["reward_core"], the value
that fixed curves are compared on, since the fence never acts on a curve
that holds the limit). The last two are the fence charge: the reactive
power the safety layer adds and the active power it curtails are charged on
top of the ordinary terms, and every step on which it acts costs a fixed
amount, so that a policy is pushed towards curves that do not need the
fence. The charge is reported as the separate term "safety" in
info["terms"].

Safety layer (EnvConfig.safety_layer): after the solve, if any feeder-1 bus
is outside the band, the InvControls are switched off, the current Q of each
unit is frozen, the sensitivity S = dV/dQ (11 x 9, pu/kvar) is measured by
finite differences (nine extra solves), the smallest additional absorption
(least squares, bounded by the remaining capability sqrt(kVA^2 - P^2) - |Q|)
that brings every violated bus to 1.05 - margin is applied and the step is
re-solved. One refinement with the same S follows; if a violation remains,
active power is curtailed with dV/dP measured the same way. The reward uses
the post-safety state: the agent proposes, the physics vetoes.
"""
import json
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import opendssdirect as dss
from scipy.optimize import lsq_linear

import cigre_dss as cd
import simulate as sim
from simulate import YearResult

# ---------------------------------------------------------------------------
# Curve parameterisation
# ---------------------------------------------------------------------------
V1, Q1, V2 = 0.92, 0.44, 0.98            # fixed points of the curve
V3_RANGE = (1.00, 1.03)
V4_GAP, V4_MAX = 0.02, 1.18              # V4 in [V3 + gap, V4_MAX]
Q4_RANGE = (0.0, 0.44)
N_UNITS = len(sim.PV_BUSES)
OBS_DIM = 6
OBS_NAMES = ["v_pu", "p_pu_kva", "q_pu_kva", "sin_tod", "cos_tod", "irr_pu"]
ACT_DIM = 3
ACT_NAMES = ["V3", "V4", "Q4"]


def _affine(a, lo, hi):
    """a in [-1, 1] -> [lo, hi]."""
    return lo + (np.clip(a, -1.0, 1.0) + 1.0) * 0.5 * (hi - lo)


def _inverse(x, lo, hi):
    return np.clip(2.0 * (x - lo) / (hi - lo) - 1.0, -1.0, 1.0)


def action_to_curve(a):
    """Action (3,) or (n, 3) in [-1, 1] -> (V3, V4, Q4) with the same leading shape."""
    a = np.asarray(a, dtype=float)
    v3 = _affine(a[..., 0], *V3_RANGE)
    v4 = _affine(a[..., 1], v3 + V4_GAP, V4_MAX)
    q4 = _affine(a[..., 2], *Q4_RANGE)
    return np.stack([v3, v4, q4], axis=-1)


def curve_to_action(v3, v4, q4):
    """Inverse of action_to_curve for scalars or arrays."""
    v3, v4, q4 = (np.asarray(x, dtype=float) for x in (v3, v4, q4))
    a0 = _inverse(v3, *V3_RANGE)
    a1 = _inverse(v4, v3 + V4_GAP, V4_MAX)
    a2 = _inverse(q4, *Q4_RANGE)
    return np.stack([a0, a1, a2], axis=-1)


def curve_xy(v3, v4, q4):
    """The four (x, y) points of the volt-var curve for one unit."""
    return (V1, V2, float(v3), float(v4)), (Q1, 0.0, 0.0, -float(q4))


DEFAULT_ACTION = curve_to_action(1.02, 1.08, 0.44)          # B1, Category B default
TIGHT_ACTION = curve_to_action(1.00, 1.04, 0.44)            # B3 of notebook 04
LAZY_ACTION = curve_to_action(1.03, 1.10, 0.44)             # B4 of notebook 04


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EnvConfig:
    pv_scale: float = 1.0            # multiplies Pmpp and kVA of every unit
    kva_ratio: float = 1.1           # inverter kVA / Pmpp (Category B oversizing)
    tr1_tap_pct: float = sim.TR1_TAP_PCT
    safety_layer: bool = True
    year: int = 2023
    seed: int = 0
    w_viol: float = 1000.0           # reward weight of the voltage excess, per pu
    w_loss: float = 0.01             # per kW of losses
    w_q: float = 0.001               # per kvar of |Q|
    w_curt: float = 0.05             # per kW of curtailed PV
    w_safety: float = 0.002          # fence charge per kvar the fence adds (and per 0.1 kW it curtails)
    w_safety_event: float = 5.0      # fence charge per step on which the fence acts
    v_hi: float = sim.V_HI
    v_lo: float = sim.V_LO
    safety_margin: float = 0.002     # the fence aims at v_hi - margin
    safety_q_cap_pu: float = None    # optional cap on the |Q| the fence may use, pu of kVA (None: capability only)
    profiles: str = "profiles_2023.h5"
    label: str = ""

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(d):
        return EnvConfig(**{k: v for k, v in d.items() if k in EnvConfig.__dataclass_fields__})


@dataclass
class EnvYearResult(YearResult):
    """YearResult whose available PV power uses the scaled Pmpp of the
    environment, so that kpi.kpis_of works unchanged at any pv_scale."""
    pv_kw: np.ndarray = None
    pv_kva: np.ndarray = None

    @property
    def p_avail_kw(self):
        p = self.irr * self.pv_kw
        kva = self.pv_kva if self.pv_kva is not None else 1.1 * self.pv_kw
        return np.where(p >= sim.CUTIN_PU * kva, p, 0.0)

    def subset(self, mask):
        return EnvYearResult(self.label, self.index[mask], self.steps[mask], self.v[mask], self.p_kw[mask],
                             self.q_kvar[mask], self.irr[mask], self.res[mask], self.com[mask],
                             self.losses_kw[mask], self.failed[mask], self.runtime_s,
                             {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v)
                              for k, v in self.extra.items()}, self.pv_kw, self.pv_kva)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class VoltVarEnv:
    """Gym-like environment (no gym dependency): reset() -> obs (9, 6);
    step(actions (9, 3)) -> obs, reward, done, info."""

    def __init__(self, config=None, pv=None, loads=None):
        self.cfg = config or EnvConfig()
        if pv is None or loads is None:
            pv, loads, _ = sim.load_year(self.cfg.profiles)
        self.index = pv.index
        self.irr_all = pv.values.astype(float)
        self.res_all = loads["residential"].values.astype(float)
        self.com_all = loads["commercial_industrial"].values.astype(float)
        self.T = len(pv)
        self.buses = list(sim.PV_BUSES)
        self.pv_kw = sim.PV_KW * self.cfg.pv_scale
        self.pv_kva = self.cfg.kva_ratio * self.pv_kw
        self.rng = np.random.default_rng(self.cfg.seed)
        self.tod = self.index.hour + self.index.minute / 60.0
        self._built = False
        self._controls_on = True
        self._steps = None
        self._k = 0
        self._prev = None          # (v, p, q) of the last solve
        self.n_solves = 0

    # ----- circuit ---------------------------------------------------------
    def build(self):
        cd.build_circuit(tr1_tap_pct=self.cfg.tr1_tap_pct)
        cd.add_pv({b: float(kw) for b, kw in zip(self.buses, self.pv_kw)},
                  kva={b: float(s) for b, s in zip(self.buses, self.pv_kva)})
        dss.Text.Command("set controlmode=static")
        dss.Text.Command("set maxcontroliter=200")
        names = dss.Circuit.AllBusNames()
        sim._node_idx = np.array([[3 * names.index(str(b)) + k for k in range(3)] for b in sim.F1_BUSES])
        x, y = curve_xy(*action_to_curve(DEFAULT_ACTION))
        for b in self.buses:
            dss.Text.Command(f"new xycurve.vv{b} npts=4 xarray=({','.join(map(str, x))}) "
                             f"yarray=({','.join(map(str, y))})")
            dss.Text.Command(f"new invcontrol.ic{b} mode=voltvar voltage_curvex_ref=rated vvc_curve1=vv{b} "
                             f"deltaq_factor=-1 refreactivepower=varmax derlist=(pvsystem.pv{b})")
        self._built = True
        self._controls_on = True

    def _set_controls(self, on):
        if on != self._controls_on:
            for b in self.buses:
                dss.Text.Command(f"invcontrol.ic{b}.enabled={'yes' if on else 'no'}")
            self._controls_on = on

    def set_curves(self, actions):
        """Write the nine curves in place (XYCurves API)."""
        curves = action_to_curve(np.asarray(actions, dtype=float).reshape(N_UNITS, ACT_DIM))
        for b, (v3, v4, q4) in zip(self.buses, curves):
            x, y = curve_xy(v3, v4, q4)
            dss.XYCurves.Name(f"vv{b}")
            dss.XYCurves.XArray(list(x))
            dss.XYCurves.YArray(list(y))
        return curves

    def _apply_profile(self, i, irr_factor=None):
        cd.scale_loads(self.res_all[i], self.com_all[i])
        irr = self.irr_all[i] if irr_factor is None else self.irr_all[i] * irr_factor
        for j, b in enumerate(self.buses):
            dss.PVsystems.Name(f"pv{b}")
            dss.PVsystems.Irradiance(float(irr[j]))

    def _solve(self, iters=3):
        """simulate.solve_step with the same three source-holding iterations
        as run_year (needed when an episode starts at an arbitrary step)."""
        self.n_solves += 1
        return sim.solve_step(iters=iters)

    def _read(self):
        v = sim.feeder1_voltages()
        p, q = sim.pv_pq()
        return v, p, q

    # ----- observations ----------------------------------------------------
    def obs_all(self, i=None):
        """(9, 6) float32: own |V| pu, own P and Q in pu of kVA (previous
        solve), sin and cos of the time of day, available irradiance pu."""
        i = self._current_step() if i is None else i
        v, p, q = self._prev
        ang = 2 * np.pi * self.tod[i] / 24.0
        obs = np.empty((N_UNITS, OBS_DIM), dtype=np.float32)
        for j, b in enumerate(self.buses):
            obs[j] = (v[sim.F1_BUSES.index(b)], p[j] / self.pv_kva[j], q[j] / self.pv_kva[j],
                      np.sin(ang), np.cos(ang), self.irr_all[i, j])
        return obs

    def global_obs(self):
        """(11,) float32: voltage of buses 1..11 of the previous solve (for a
        later feeder-level coordinator; unused by the local agents)."""
        return self._prev[0].astype(np.float32)

    # ----- episode API -----------------------------------------------------
    def _current_step(self):
        return int(self._steps[min(self._k, len(self._steps) - 1)])

    def reset(self, day=None, steps=None):
        """Start an episode: `day` (0..364) -> the 96 steps of that day;
        `steps` -> an explicit array of step indices; neither -> the whole
        year. The state of the step before the first one is obtained with the
        default curve, so that the first observation carries a sensible
        'previous' V, P and Q. Returns obs (9, 6)."""
        self.build()          # always: OpenDSS holds one circuit, another environment may have used it
        if steps is not None:
            self._steps = np.asarray(steps, dtype=int)
        elif day is not None:
            self._steps = np.arange(day * 96, (day + 1) * 96)
        else:
            self._steps = np.arange(self.T)
        self._k = 0
        self._set_controls(True)
        self.set_curves(np.tile(DEFAULT_ACTION, (N_UNITS, 1)))
        self._apply_profile(max(int(self._steps[0]) - 1, 0))
        self._solve(iters=3)
        self._prev = self._read()
        return self.obs_all()

    def step(self, actions, control=True):
        """Apply the nine curves (actions (9, 3) in [-1, 1]) to the current
        step, solve, run the safety layer if configured, and advance.
        control=False disables the curves and fixes Q = 0 (unity power
        factor, used for the B0 reference). Returns obs, reward, done, info."""
        cfg = self.cfg
        i = self._current_step()
        self._apply_profile(i)
        if control:
            self._set_controls(True)
            curves = self.set_curves(actions)
        else:
            self._set_controls(False)
            sim.set_pv_q(np.zeros(N_UNITS))
            curves = None
        failed = self._solve()
        v, p, q = self._read()
        p_avail = self.irr_all[i] * self.pv_kw
        p_avail = np.where(p_avail >= sim.CUTIN_PU * self.pv_kva, p_avail, 0.0)
        info = dict(step=i, time=self.index[i], curves=curves, failed=failed,
                    safety_active=False, safety_dq=0.0, safety_dp=0.0, safety_solves=0,
                    v_before=v.copy(), q_before=q.copy(), p_before=p.copy())
        if cfg.safety_layer and self._violated(v):
            v, p, q = self._safety(v, p, q, p_avail, info)
        loss = sim.losses_kw()
        excess = np.maximum(0.0, np.maximum(v - cfg.v_hi, cfg.v_lo - v)).sum()
        curt = np.maximum(0.0, p_avail - p).sum()
        terms = dict(viol=-cfg.w_viol * excess, loss=-cfg.w_loss * loss,
                     q=-cfg.w_q * np.abs(q).sum(), curt=-cfg.w_curt * curt,
                     safety=-(cfg.w_safety * (info["safety_dq"] + 10.0 * info["safety_dp"])
                              + cfg.w_safety_event * float(info["safety_active"])))
        reward = float(sum(terms.values()))
        info.update(v=v, p_kw=p, q_kvar=q, p_avail_kw=p_avail, losses_kw=loss, terms=terms,
                    excess_pu=excess, curtailed_kw=curt, reward_core=reward - terms["safety"])
        self._prev = (v, p, q)
        self._k += 1
        done = self._k >= len(self._steps)
        return self.obs_all(), reward, done, info

    def random_action(self):
        return self.rng.uniform(-1.0, 1.0, size=(N_UNITS, ACT_DIM))

    # ----- safety layer ----------------------------------------------------
    def _violated(self, v):
        return bool((v > self.cfg.v_hi).any() or (v < self.cfg.v_lo).any())

    def _sensitivity(self, q, p, kind="q", h_pu=0.01):
        """Finite-difference S = dV/dQ (kind 'q', perturbation -h_pu kVA of
        absorption) or dV/dP (kind 'p', perturbation -h_pu Pmpp of output),
        11 x 9 in pu per kvar or per kW, at the current fixed point."""
        S = np.empty((len(sim.F1_BUSES), N_UNITS))
        dss.Solution.Solve(); self.n_solves += 1
        v0 = sim.feeder1_voltages()
        for j, b in enumerate(self.buses):
            dss.PVsystems.Name(f"pv{b}")
            if kind == "q":
                h = h_pu * self.pv_kva[j]
                dss.PVsystems.kvar(float(q[j] - h))
            else:
                h = h_pu * self.pv_kw[j]
                irr0 = dss.PVsystems.Irradiance()
                dss.PVsystems.Irradiance(max(irr0 - h / self.pv_kw[j], 0.0))
            dss.Solution.Solve(); self.n_solves += 1
            S[:, j] = (v0 - sim.feeder1_voltages()) / h        # dV per unit of injection
            dss.PVsystems.Name(f"pv{b}")
            if kind == "q":
                dss.PVsystems.kvar(float(q[j]))
            else:
                dss.PVsystems.Irradiance(irr0)
        return S

    def sensitivity(self, kind="q"):
        """The sensitivity matrix the fence would measure at the operating
        point of the last solve (InvControls off, Q frozen at its current
        value): dV/dQ (kind 'q', pu per kvar) or dV/dP (kind 'p', pu per kW),
        11 feeder-1 buses x 9 units."""
        v, p, q = self._prev
        self._set_controls(False)
        sim.set_pv_q(q)
        return self._sensitivity(q, p, kind)

    def _required_drop(self, v):
        """Signed voltage change needed at each bus (negative = must fall)."""
        cfg = self.cfg
        target = np.zeros_like(v)
        over = v > cfg.v_hi; under = v < cfg.v_lo
        target[over] = (cfg.v_hi - cfg.safety_margin) - v[over]
        target[under] = (cfg.v_lo + cfg.safety_margin) - v[under]
        return target, over | under

    def _solve_lsq(self, S, v, lo, hi):
        """Least-squares change x (per unit) with S x ~ required drop on the
        violated buses only, within [lo, hi]."""
        target, rows = self._required_drop(v)
        x = np.zeros(N_UNITS)
        free = hi > lo + 1e-9                  # units that still have room to move
        if not rows.any() or not free.any():
            return x
        res = lsq_linear(S[rows][:, free], target[rows], bounds=(lo[free], hi[free]),
                         method="bvls", tol=1e-12)
        x[free] = res.x
        return x

    def _safety(self, v, p, q, p_avail, info):
        cfg = self.cfg
        info["safety_active"] = True
        i = info["step"]
        self._set_controls(False)
        sim.set_pv_q(q)
        n0 = self.n_solves
        # 1. reactive power: S = dV/dQ, then the bounded least-squares correction, twice
        S = self._sensitivity(q, p, "q")
        qcap = np.sqrt(np.maximum(self.pv_kva ** 2 - p ** 2, 0.0))
        if cfg.safety_q_cap_pu is not None:
            qcap = np.minimum(qcap, cfg.safety_q_cap_pu * self.pv_kva)
        q_set = q.copy()                      # the kvar setpoint currently applied
        for _ in range(2):                    # correction + one refinement with the same S
            only_over = not (v < cfg.v_lo).any()
            only_under = not (v > cfg.v_hi).any()
            lo = -qcap - q_set
            hi = qcap - q_set
            if only_over:
                hi = np.zeros(N_UNITS)          # absorption only
            elif only_under:
                lo = np.zeros(N_UNITS)          # injection only
            lo = np.minimum(lo, 0.0); hi = np.maximum(hi, 0.0)
            x = self._solve_lsq(S, v, lo, hi)
            if not np.any(x):
                break
            q_set = q_set + x
            sim.set_pv_q(q_set)
            self._solve()
            v, p, q = self._read()
            if not self._violated(v):
                break
        info["safety_dq"] = float(np.abs(q_set - info["q_before"]).sum())
        # 2. active power, only if reactive power was not enough
        if self._violated(v):
            Sp = self._sensitivity(q_set, p, "p")
            lo = -p.copy(); hi = np.zeros(N_UNITS)
            x = self._solve_lsq(Sp, v, np.minimum(lo, 0.0), hi)
            if np.any(x):
                factor = np.clip((p + x) / np.where(p > 0, p, 1.0), 0.0, 1.0)
                self._apply_profile(i, irr_factor=np.where(p > 0, factor, 1.0))
                sim.set_pv_q(q_set)
                self._solve()
                v, p, q = self._read()
                info["safety_dp"] = float(np.maximum(0.0, info["p_before"] - p).sum())
        info["safety_solves"] = self.n_solves - n0
        return v, p, q

    # ----- convenience: a whole run in YearResult form -------------------
    def run_policy(self, policy_fn, steps=None, label="", verbose=True):
        """Run `steps` (default: the whole year) with policy_fn(obs, env) ->
        actions (9, 3); policy_fn=None means unity power factor without any
        curve (B0). Returns an EnvYearResult (kpi.kpis_of works on it) with
        the reward, its terms and the safety statistics in .extra."""
        obs = self.reset(steps=steps if steps is not None else np.arange(self.T))
        n = len(self._steps)
        v = np.empty((n, len(sim.F1_BUSES))); p = np.empty((n, N_UNITS)); q = np.empty((n, N_UNITS))
        loss = np.empty(n); failed = np.zeros(n, dtype=bool); rew = np.empty(n)
        terms = {k: np.empty(n) for k in ("viol", "loss", "q", "curt", "safety")}
        s_act = np.zeros(n, dtype=bool); s_dq = np.zeros(n); s_dp = np.zeros(n); s_ns = np.zeros(n, dtype=int)
        curves = np.full((n, N_UNITS, ACT_DIM), np.nan)
        v_before = np.empty((n, len(sim.F1_BUSES)))
        n_solve0 = self.n_solves
        t0 = time.perf_counter()
        for k in range(n):
            if policy_fn is None:
                obs, r, done, info = self.step(None, control=False)
            else:
                obs, r, done, info = self.step(policy_fn(obs, self))
            v[k] = info["v"]; p[k] = info["p_kw"]; q[k] = info["q_kvar"]; loss[k] = info["losses_kw"]
            failed[k] = info["failed"]; rew[k] = r
            for t in terms:
                terms[t][k] = info["terms"][t]
            s_act[k] = info["safety_active"]; s_dq[k] = info["safety_dq"]; s_dp[k] = info["safety_dp"]
            s_ns[k] = info["safety_solves"]; v_before[k] = info["v_before"]
            if info["curves"] is not None:
                curves[k] = info["curves"]
        elapsed = time.perf_counter() - t0
        st = self._steps
        if verbose:
            print(f"{label or 'run':34s}: {n} steps in {elapsed:6.1f} s ({elapsed / n * 1e3:.2f} ms per step), "
                  f"safety active on {int(s_act.sum())} steps, {self.n_solves - n_solve0} solves, "
                  f"control-loop failures {int(failed.sum())}")
        extra = dict(reward=rew, term_viol=terms["viol"], term_loss=terms["loss"], term_q=terms["q"],
                     term_curt=terms["curt"], term_safety=terms["safety"], reward_core=rew - terms["safety"],
                     safety_active=s_act, safety_dq=s_dq, safety_dp=s_dp,
                     safety_solves=s_ns, v_before=v_before, curves=curves, config=asdict(self.cfg))
        return EnvYearResult(label, self.index[st], st, v, p, q, self.irr_all[st], self.res_all[st],
                             self.com_all[st], loss, failed, elapsed, extra, self.pv_kw, self.pv_kva)


def constant_policy(action):
    """policy_fn that applies the same action to all nine units."""
    a = np.tile(np.asarray(action, dtype=float), (N_UNITS, 1))
    return lambda obs, env: a


def random_policy(obs, env):
    return env.random_action()


def reward_totals(result):
    """Sum of the reward and of each term over a run."""
    e = result.extra
    safety = float(e["term_safety"].sum()) if "term_safety" in e else 0.0
    return dict(total=float(e["reward"].sum()), viol=float(e["term_viol"].sum()), loss=float(e["term_loss"].sum()),
                q=float(e["term_q"].sum()), curt=float(e["term_curt"].sum()), safety=safety,
                core=float(e["reward"].sum()) - safety)


def reward_core_of(result, mask=None):
    """Sum of the core reward (the four ordinary terms, without the fence
    charge) over a run, or over the steps where mask is True."""
    e = result.extra
    r = e["reward_core"] if "reward_core" in e else e["reward"] - e.get("term_safety", 0.0)
    return float(r.sum() if mask is None else r[mask].sum())
