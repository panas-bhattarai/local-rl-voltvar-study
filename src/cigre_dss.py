"""CIGRE European MV benchmark (CIGRE TB 575, Sec. 6.2) in OpenDSS, with a
pandapower reference load flow built from the same data file.

Data file: data/cigre_mv_european_tb575.json, the parameter extraction of the
CIGRE-MV-PSCAD repository (https://github.com/panas-bhattarai/CIGRE-MV-PSCAD),
copied here unchanged. Page numbers quoted in the comments are PDF page
indices of the brochure, as used in that file.

Modelling conventions (mirroring scripts/validate_cigre_mv.py of that repo):
  * bus 0 is the 110 kV slack, held at exactly 110 kV, 0 deg (Table 9.6, p.103);
  * TR1/TR2: 25 MVA 110/20 kV, 0.016 + j1.92 ohm on the 20 kV side (Table 6.13,
    p.54) = 0.1 % + j12 % on 25 MVA / 20 kV; LV tap +6.25 % (TR1) and +3.125 %
    (TR2) from Table 9.7 (p.104), impedance referred through the tapped ratio;
    30 deg shift with the 20 kV side leading, as in Table 9.6;
  * lines from Table 6.12 (p.54), positive and zero sequence, B in uS/km
    converted to nF/km; open tie lines are removed from the model;
  * loads from Table 6.15 (p.55), constant power (P, Q from S and pf).
"""
import json
import math
import os

import opendssdirect as dss

# Local copy of cigre_mv_european_tb575.json, credit: CIGRE-MV-PSCAD repository
# (https://github.com/panas-bhattarai/CIGRE-MV-PSCAD), data/ folder.
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data",
                         "cigre_mv_european_tb575.json")
W = 2 * math.pi * 50.0          # rad/s, 50 Hz system (p.51)

# PV ratings (kW) per bus. Source: R. Wagle, P. Sharma, M. Amin, J. L. Rueda,
# F. Gonzalez-Longatt, "Real-time Volt-Var control of grid forming converters
# in DER-enriched distribution network", Front. Energy Res. 10:1054870 (2023),
# Table 1 (attributed there to Barsali et al., 2014). The nine buses are those
# that carry PV or wind in TB 575 Table 6.18 (p.58); the 1.5 MW wind unit at
# bus 7 is replaced by PV so that all units are PV. Only Pmax is tabulated; no
# kVA or kvar rating is given.
PV_KW = {3: 690, 4: 690, 5: 680, 6: 680, 7: 740, 8: 740, 9: 740, 10: 740, 11: 740}

# Base (unscaled) kW/kvar of every Load element created by build_circuit, keyed
# by element name; used by scale_loads().
_LOAD_BASE = {}


def load_data(data_path=DATA_PATH):
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


def build_circuit(data_path=DATA_PATH, closed_switches=(), load_scale=1.0,
                  tr1_tap_pct=None):
    """Clear OpenDSS and build the feeder. Returns nothing; call solve() next.

    closed_switches: iterable of switch ids ("S1", "S2", "S3") whose tie line
    is enabled; all open by default (radial base case, p.103).
    tr1_tap_pct: LV-side tap of TR1 in per cent. None (default) keeps the
    brochure value of Table 9.7 (+6.25 %). TR2 always keeps its brochure tap."""
    d = load_data(data_path)
    cmd = dss.Text.Command
    cmd("clear")
    # 50 Hz system (p.51). Must precede 'new circuit': every element takes the
    # default base frequency at creation, and a Vsource whose base frequency
    # differs from the solution frequency is treated as a harmonic source.
    cmd(f"set defaultbasefrequency={d['system']['frequency_Hz']}")

    # 110 kV grid equivalent: 5000 MVA, R/X = 0.1 (Table 6.14, p.54).
    hv = d["hv_equivalent"]["network_specific"]
    xr = 1.0 / hv["R_over_X"]
    cmd(f"new circuit.cigre_mv basekv={hv['nominal_voltage_kV']} pu=1.0 angle=0 "
        f"bus1=0 phases=3 MVAsc3={hv['short_circuit_power_MVA']} "
        f"MVAsc1={hv['short_circuit_power_MVA']} x1r1={xr} x0r0={xr}")

    # Transformers (Table 6.13, p.54; taps Table 9.7, p.104). Dyn with the
    # 20 kV side leading by 30 deg (leadlag=lead) to follow Table 9.6.
    for tr in d["transformers"]:
        R, X = tr["Z_ohm_ref_V2"]["R"], tr["Z_ohm_ref_V2"]["X"]
        zb = tr["V2_kV"] ** 2 / tr["S_rated_MVA"]           # 16 ohm
        r_pct, x_pct = R / zb * 100, X / zb * 100             # 0.1 %, 12 %
        tap_lv_pct = tr["tap_setting_used_in_power_flow"]["secondary_pct"]
        if tr1_tap_pct is not None and tr["id"].upper() == "TR1":
            tap_lv_pct = tr1_tap_pct
        tap_lv = 1 + tap_lv_pct / 100
        tap_hv = 1 + tr["tap_setting_used_in_power_flow"]["primary_pct"] / 100
        cmd(f"new transformer.{tr['id']} phases=3 windings=2 xhl={x_pct} "
            f"leadlag=lead %noloadloss=0 %imag=0 "
            f"wdg=1 bus={tr['from_bus']} conn=delta kv={tr['V1_kV']} "
            f"kva={tr['S_rated_MVA'] * 1000} %r={r_pct / 2} tap={tap_hv} "
            f"wdg=2 bus={tr['to_bus']} conn=wye kv={tr['V2_kV']} "
            f"kva={tr['S_rated_MVA'] * 1000} %r={r_pct / 2} tap={tap_lv}")

    # Lines (Table 6.12, p.54). Switch S1/S2/S3 sit in series with segments
    # 15, 6, 11 (Fig. 6.5, p.52); all open in the radial base case (p.103).
    open_segments = {s["line_segment"] for s in d["switches"]
                     if s["id"] not in closed_switches}
    for l in d["lines"]:
        c1 = l["B1_uS_per_km"] / W * 1e3                      # uS/km -> nF/km
        c0 = l["B0_uS_per_km"] / W * 1e3
        en = "no" if l["segment"] in open_segments else "yes"
        cmd(f"new line.seg{l['segment']} phases=3 bus1={l['from_bus']} "
            f"bus2={l['to_bus']} length={l['length_km']} units=km "
            f"r1={l['R1_ohm_per_km']} x1={l['X1_ohm_per_km']} c1={c1:.6f} "
            f"r0={l['R0_ohm_per_km']} x0={l['X0_ohm_per_km']} c0={c0:.6f} "
            f"enabled={en}")

    # Loads (Table 6.15, p.55): constant power (model=1). vminpu lowered so
    # that OpenDSS never switches a load to constant impedance. One Load
    # element per bus and sector (b<bus>_res, b<bus>_com) so that the two
    # sectors can be scaled separately with the daily profiles of Figure 6.4.
    _LOAD_BASE.clear()
    for ld in d["loads"]:
        for sec in ("residential", "commercial_industrial"):
            sd = ld[sec]
            if sd["S_kVA"]:
                name = f"b{ld['bus']}_{sec[:3]}"
                kw, kvar = sd["P_kW_derived"] * load_scale, sd["Q_kvar_derived"] * load_scale
                _LOAD_BASE[name] = (kw, kvar)
                cmd(f"new load.{name} bus1={ld['bus']} phases=3 "
                    f"conn=wye kv={d['system']['nominal_voltage_kV']} model=1 "
                    f"kw={kw} kvar={kvar} vminpu=0.5 vmaxpu=1.5")

    cmd("set voltagebases=[110 20]")
    cmd("calcvoltagebases")
    cmd("set mode=snapshot")
    cmd("set tolerance=1e-8")
    cmd("set maxiterations=100")


def solve(bus0_kv=110.0, iters=6):
    """Solve, adjusting the source EMF magnitude and angle so that bus 0 sits
    at exactly bus0_kv, 0 deg (the brochure and pandapower treat bus 0 as the
    slack bus; the 5000 MVA equivalent then only matters for short-circuit
    studies)."""
    dss.Solution.Solve()
    for _ in range(iters):
        b0 = bus_voltages_kv()[0]
        dss.Vsources.First()
        dss.Vsources.PU(dss.Vsources.PU() * bus0_kv / b0["v_kv"])
        dss.Vsources.AngleDeg(dss.Vsources.AngleDeg() - b0["ang"])
        dss.Solution.Solve()
    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS did not converge")


def bus_voltages_kv():
    """dict bus -> {'v_kv': line-to-line kV (mean of phases), 'ang': phase-A angle deg}."""
    res = {}
    for name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(name)
        va = dss.Bus.VMagAngle()            # [|Va|, angA, |Vb|, angB, |Vc|, angC] (V, deg)
        mags = va[0::2]
        res[int(name)] = dict(v_kv=sum(mags) / len(mags) * math.sqrt(3) / 1e3,
                              ang=va[1])
    return dict(sorted(res.items()))


def bus_voltages_pu():
    """dict bus -> voltage in pu of the bus base (110 kV for bus 0, 20 kV otherwise)."""
    return {b: r["v_kv"] / (110.0 if b == 0 else 20.0) for b, r in bus_voltages_kv().items()}


def transformer_results():
    """dict id -> LV current (A, mean of phases) and HV-side P/Q (MW, Mvar)."""
    out = {}
    for name in dss.Transformers.AllNames():
        dss.Circuit.SetActiveElement(f"transformer.{name}")
        n = dss.CktElement.NumConductors()
        cm = dss.CktElement.CurrentsMagAng()
        i_lv = [cm[2 * (n + k)] for k in range(3)]             # terminal 2, phases a,b,c
        p = dss.CktElement.Powers()                             # kW,kvar per conductor
        p_hv = sum(p[2 * k] for k in range(n)) / 1e3
        q_hv = sum(p[2 * k + 1] for k in range(n)) / 1e3
        out[name.upper()] = dict(i_lv_a=sum(i_lv) / 3, p_hv_mw=p_hv, q_hv_mvar=q_hv)
    return out


def grid_pq():
    """P (MW) and Q (Mvar) delivered by the 110 kV source at bus 0."""
    p, q = dss.Circuit.TotalPower()
    return -p / 1e3, -q / 1e3


def line_pq(segment):
    """P (MW), Q (Mvar) entering line seg<segment> at its bus1 end; positive
    means flow from bus1 towards bus2, negative means reverse flow."""
    dss.Circuit.SetActiveElement(f"line.seg{segment}")
    p = dss.CktElement.Powers()
    n = dss.CktElement.NumConductors()
    return sum(p[2 * k] for k in range(n)) / 1e3, sum(p[2 * k + 1] for k in range(n)) / 1e3


# --------------------------------------------------------------------------
# Loads through the day and PV units
# --------------------------------------------------------------------------

def scale_loads(res_pu, ci_pu):
    """Scale every residential load to res_pu and every commercial/industrial
    load to ci_pu times its base value (P and Q together, i.e. constant pf),
    as the brochure profiles of Figure 6.4 are in pu of peak apparent power."""
    for name, (kw, kvar) in _LOAD_BASE.items():
        f = res_pu if name.endswith("_res") else ci_pu
        dss.Loads.Name(name)
        dss.Loads.kW(kw * f)
        dss.Loads.kvar(kvar * f)


def add_pv(units, kva=None, pf=1.0, irradiance=1.0, kva_ratio=1.1):
    """Add one three-phase PVSystem per entry of `units` (dict bus -> kW).

    kva: None -> inverter rating kva_ratio * kW per unit (default 1.1, the
         oversizing that gives the Category B reactive capability of
         IEEE 1547-2018 at full active output); a number -> same kVA for all
         units; a dict bus -> kVA.
    pf:  fixed power factor (1.0 = unity, no reactive support). Positive =
         injecting vars (OpenDSS convention for generators), negative = absorbing.
    The units are constant-power over 0.5..1.5 pu (vminpu/vmaxpu) so that they
    are never switched to constant impedance in the overvoltage cases; the
    default 1.1 pu ceiling would otherwise soften the very effect under study.
    Pmpp is quoted at 1 kW/m2 and 25 C with no temperature or efficiency
    derating, so P = irradiance * Pmpp (limited by kVA).
    """
    cmd = dss.Text.Command
    for bus, kw in units.items():
        if kva is None:
            s = kva_ratio * kw
        elif isinstance(kva, dict):
            s = kva[bus]
        else:
            s = kva
        cmd(f"new pvsystem.pv{bus} phases=3 bus1={bus} conn=wye kv=20 "
            f"kva={s} pmpp={kw} irradiance={irradiance} pf={pf} "
            f"%cutin=0.1 %cutout=0.1 vminpu=0.5 vmaxpu=1.5 model=1")


def set_pv_irradiance(x):
    """Set the irradiance (pu of 1 kW/m2) of every PVSystem in the circuit."""
    if dss.PVsystems.Count() == 0:
        return
    dss.PVsystems.First()
    while True:
        dss.PVsystems.Irradiance(x)
        if not dss.PVsystems.Next():
            break


def set_pv_irradiances(by_bus):
    """Set the irradiance of each PVSystem individually: by_bus is a dict
    bus -> pu of 1 kW/m2."""
    for bus, x in by_bus.items():
        dss.PVsystems.Name(f"pv{bus}")
        dss.PVsystems.Irradiance(float(x))


def pv_results():
    """dict bus -> {'p_kw': injected P, 'q_kvar': injected Q} of each PVSystem."""
    out = {}
    if dss.PVsystems.Count() == 0:
        return out
    dss.PVsystems.First()
    while True:
        name = dss.PVsystems.Name()
        dss.Circuit.SetActiveElement(f"pvsystem.{name}")
        p = dss.CktElement.Powers()                 # kW, kvar per conductor, load sign
        n = dss.CktElement.NumConductors()
        out[int(name[2:])] = dict(p_kw=-sum(p[2 * k] for k in range(n)),
                                  q_kvar=-sum(p[2 * k + 1] for k in range(n)))
        if not dss.PVsystems.Next():
            break
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------
# pandapower reference
# --------------------------------------------------------------------------

def pandapower_reference(d, load_scale=1.0, closed_switches=(), pv_kw=None,
                         pv_irradiance=1.0, sector_scale=(1.0, 1.0), tr1_tap_pct=None):
    """Reference load flow (adapted from CIGRE-MV-PSCAD/scripts/validate_cigre_mv.py).
    pv_kw: optional dict bus -> kW of unity-pf static generators (PV at
    pv_irradiance pu); sector_scale = (residential pu, commercial/industrial pu);
    tr1_tap_pct as in build_circuit.
    Returns (bus dict like bus_voltages_kv, transformer dict, grid P MW, grid Q Mvar)."""
    import pandapower as pp
    net = pp.create_empty_network(f_hz=50, sn_mva=25)
    b = {0: pp.create_bus(net, 110, name="Bus 0")}
    for i in range(1, 15):
        b[i] = pp.create_bus(net, 20, name="Bus %d" % i)
    hv = d["hv_equivalent"]["network_specific"]
    pp.create_ext_grid(net, b[0], vm_pu=1.0, va_degree=0.0,
                       s_sc_max_mva=hv["short_circuit_power_MVA"], rx_max=hv["R_over_X"])
    for tr in d["transformers"]:
        R, X = tr["Z_ohm_ref_V2"]["R"], tr["Z_ohm_ref_V2"]["X"]
        zb = tr["V2_kV"] ** 2 / tr["S_rated_MVA"]
        tap = tr["tap_setting_used_in_power_flow"]["secondary_pct"]
        if tr1_tap_pct is not None and tr["id"].upper() == "TR1":
            tap = tr1_tap_pct
        pp.create_transformer_from_parameters(
            net, b[tr["from_bus"]], b[tr["to_bus"]], sn_mva=tr["S_rated_MVA"],
            vn_hv_kv=tr["V1_kV"], vn_lv_kv=tr["V2_kV"], vkr_percent=R / zb * 100,
            vk_percent=math.hypot(R, X) / zb * 100, pfe_kw=0, i0_percent=0,
            shift_degree=-30, tap_side="lv", tap_neutral=0, tap_min=-16, tap_max=16,
            tap_step_percent=0.625, tap_pos=round(tap / 0.625),
            tap_changer_type="Ratio", name=tr["id"])
    sw = {s["line_segment"] for s in d["switches"] if s["id"] not in closed_switches}
    for l in d["lines"]:
        pp.create_line_from_parameters(
            net, b[l["from_bus"]], b[l["to_bus"]], l["length_km"], l["R1_ohm_per_km"],
            l["X1_ohm_per_km"], l["B1_uS_per_km"] / W * 1e3, max_i_ka=0.2,
            r0_ohm_per_km=l["R0_ohm_per_km"], x0_ohm_per_km=l["X0_ohm_per_km"],
            c0_nf_per_km=l["B0_uS_per_km"] / W * 1e3, name="Seg %d" % l["segment"],
            in_service=(l["segment"] not in sw))
    for ld in d["loads"]:
        for sec, f in zip(("residential", "commercial_industrial"), sector_scale):
            sd = ld[sec]
            if sd["S_kVA"]:
                pp.create_load(net, b[ld["bus"]], p_mw=sd["P_kW_derived"] / 1e3 * load_scale * f,
                               q_mvar=sd["Q_kvar_derived"] / 1e3 * load_scale * f)
    for bus, kw in (pv_kw or {}).items():
        pp.create_sgen(net, b[bus], p_mw=kw / 1e3 * pv_irradiance, q_mvar=0.0, name=f"pv{bus}")
    pp.runpp(net, init="flat", tolerance_mva=1e-10)
    res = {i: dict(v_kv=float(net.res_bus.vm_pu[b[i]] * net.bus.vn_kv[b[i]]),
                   ang=float(net.res_bus.va_degree[b[i]])) for i in range(15)}
    trf = {tr["id"]: dict(i_lv_a=float(net.res_trafo.i_lv_ka[k] * 1e3),
                          p_hv_mw=float(net.res_trafo.p_hv_mw[k]),
                          q_hv_mvar=float(net.res_trafo.q_hv_mvar[k]))
           for k, tr in enumerate(d["transformers"])}
    return res, trf, float(net.res_ext_grid.p_mw[0]), float(net.res_ext_grid.q_mvar[0])
