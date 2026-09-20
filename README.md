# Local reinforcement learning against fixed volt-var curves on a CIGRE MV feeder

A reproducible study of how far **local** control of PV inverters can go in keeping a
distribution feeder inside its voltage limits, and where it stops.

Nine PV units (6.44 MW) on the CIGRE European MV benchmark feeder push the far buses above
1.05 pu for about 200 hours of a simulated year. Every unit can hold that limit with the
volt-var curve of IEEE 1547-2018, but the standard's default curve is deliberately cautious
and pays for it in reactive energy and losses. The question here is whether an agent at each
inverter, seeing only its own bus, can learn to set its curve better, and how close that gets
to what a feeder-wide optimiser would do.

**Short answer: it beats the default curve comfortably, and it does not beat a well-chosen
fixed curve.** The information that is missing is not local.

## Results

Full simulated year, 15-minute steps, nine inverters, all controllers inside the adjustable
ranges of IEEE 1547 Category B:

| controller | hours outside 0.95-1.05 pu | losses (MWh) | reactive energy absorbed (Mvarh) | score |
|---|---|---|---|---|
| unity power factor | 199.75 | 383.7 | 0 | -31,328 |
| IEEE 1547 default curve (V3 1.02, V4 1.08) | 0 | 395.5 | 1,040 | -20,147 |
| tight curve (1.00, 1.04) | 0 | 453.1 | 4,272 | -35,380 |
| lazy curve (1.03, 1.10) | 0 | 387.4 | 294 | -16,836 |
| **best fixed curve + safety layer (1.03, 1.16)** | 0 | 385.8 | 202 | **-16,405** |
| learned, seed 0 / 1 / 2 | 0 | 386.4 / 385.6 / 386.6 | 372 / 310 / 266 | -17,112 / -16,828 / -16,695 |

The score is the objective the agents are trained on (voltage excess, losses, reactive power,
curtailment), higher is better. Details in `07_results.ipynb`.

Findings, in order of how much they surprised us:

1. **A fixed curve wins.** Sweeping 15 fixed curves and giving each the same safety layer the
   agents use, the mildest one (V3 1.03, V4 1.16) beats all three learned policies. The
   time-varying curves the agents learn do not reach a loss and reactive-energy combination
   that a fixed curve cannot.
2. **The limit needs its own guard.** With the safety layer switched off, the learned
   policies leave the band for 9.5 to 33 hours a year, and so do the milder fixed curves.
   Under this reward a short, small breach is cheaper than the reactive power it saves, so
   the limit cannot be left to the objective alone.
3. **Local observation cannot allocate.** A per-step OPF that sees the whole feeder puts 85 %
   of its absorption at the three far buses. Every local controller spreads it out. A unit
   cannot tell from its own voltage whether the feeder is near its limit, or whether its own
   kvar is the one that matters.
4. **On an ordinary sunny day the right action is nothing.** The OPF does nothing and scores
   exactly the no-control result; every curve, fixed or learned, pays a little for insurance.
5. **The results survive an unseen year.** Repeating everything on a second year of profiles
   from a different seed leaves the ranking unchanged and every margin within 0.2 points.
6. **Inverters at night matter.** The simulated inverters keep their reactive capability at
   zero PV output, so a curve whose knee sits below the night-time voltage absorbs all night:
   479 of the default curve's 1,040 Mvarh, and 58 to 174 Mvarh of the learned policies'.

## What is in here

Seven notebooks, one step each, meant to be read in order:

| notebook | what it does |
|---|---|
| `01_feeder.ipynb` | the CIGRE European MV feeder in OpenDSS, validated against pandapower and the source brochure |
| `02_pv.ipynb` | nine PV units, the transformer tap, where and when overvoltage appears |
| `03_profiles.ipynb` | one synthetic year of PV and load at 15 minutes, and a screening of every step |
| `04_baselines.ipynb` | fixed volt-var curves, volt-watt, and a per-step OPF as reference |
| `05_environment.ipynb` | the learning environment: observations, actions, reward, safety layer |
| `06_train.ipynb` | soft actor-critic with one shared actor for nine units, three seeds |
| `07_results.ipynb` | all of it compared, plus the fixed-curve sweep, the safety-layer test and the unseen year |

Modules: `cigre_dss.py` (feeder), `profiles.py` (PV and load models), `simulate.py` (year
runs, OPF), `kpi.py` (metrics), `env.py` (environment and safety layer), `agents.py` (SAC).
Figures are written to `figures/`, headline numbers to `results_summary.json`.

## Method in brief

- **Network.** CIGRE European MV benchmark (CIGRE Technical Brochure 575, Section 6.2),
  feeder 1, 20 kV, 50 Hz, in OpenDSS through `opendssdirect.py`. The transformer tap is fixed
  at +4.375 %, chosen so that the feeder without PV stays inside the band all year.
- **PV.** Nine units of 680 to 740 kW at buses 3 to 11, inverters rated 1.1 times Pmpp, as in
  published studies of this feeder.
- **Profiles.** Synthetic and seeded, with no downloads: clear-sky geometry with a two-state
  Markov cloud model, and the brochure's daily load shapes with seasonal, weekly and noise
  factors.
- **Control.** Each unit sets the upper half of its IEEE 1547 volt-var curve (V3, V4, Q4)
  every 15 minutes, within the standard's adjustable ranges. The lower half is fixed.
- **Agents.** Soft actor-critic, one actor and one critic pair shared by the nine units, with
  a one-hot unit identity appended to six local observations. Execution stays local.
- **Safety layer.** Before a step is scored, if any bus would leave the band, a
  sensitivity-based correction adds the least extra absorption that brings it back, and
  curtails active power only if that is not enough. Its use is charged in the reward.
- **References.** Fixed curves (default, tight, lazy, and a 15-point sweep) and a per-step OPF
  that knows every bus voltage, run with the same objective as the agents.

## Reproducing

```bash
pip install opendssdirect.py numpy pandas scipy matplotlib h5py torch pandapower jupyter
```

Run the notebooks in order; each writes the files the next one reads. Tested with Python
3.11, opendssdirect.py 0.9.4 (DSS C-API 0.14.5), numpy 2.4.6, pandas 2.3.3, scipy 1.16.3,
h5py 3.16, matplotlib 3.11, torch 2.14 (CPU) and pandapower 3.5.4, the last used only to
validate the feeder in notebook 01.

Everything runs on a CPU. Notebook 06 takes about 30 minutes for three seeds and notebook 07
about 14 minutes; the rest take minutes or less. The large intermediate result files
(`train_2023.h5`, `baselines_2023.h5`, `env_checks_2023.h5`) are not tracked here: they are
regenerated by running notebooks 04, 05 and 06. The trained policies are in `models/`.

## Limitations

Synthetic profiles from a simple weather model; one feeder and one PV penetration; balanced
positive-sequence power flow every 15 minutes, with no dynamics, unbalance, measurement noise
or communication delay; a fixed transformer tap and no other voltage-control device; a safety
layer that uses the exact network model; three training seeds, whose spread is as wide as some
of the differences being compared; and one particular choice of reward weights, whose voltage
term is weak enough to rank some limit-breaking curves highly. Section 13 of
`07_results.ipynb` states these in full.

## Where this goes next

The gap that local control cannot close is about the feeder, not the inverter: whether the
limit is at risk at all right now, and which units move the critical bus. Both are visible at
the substation. A slow feeder-level coordinator could pass them down as a small signal, such
as a per-unit budget, a shift of the curve's knee, or simply "act now" and "stand down", while
the fast decisions stay local and the shape of the standard's curve is kept. Whether such a
coordinator closes the remaining gap, and how it behaves when its messages are delayed or
lost, is the natural continuation.

## License

MIT, see `LICENSE`. The benchmark data in `data/` are derived from the published CIGRE
Technical Brochure 575 description of the European MV network.

## Citation

Bhattarai, P. (2026). *Local reinforcement learning against fixed volt-var curves on a CIGRE
MV feeder.* Software and notebooks.
