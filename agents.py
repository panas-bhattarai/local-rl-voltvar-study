"""Soft actor-critic (SAC) with a shared actor for the nine PV units.

Algorithm (Haarnoja et al. 2018, "Soft actor-critic: off-policy maximum
entropy deep reinforcement learning with a stochastic actor", ICML; and the
automatic entropy tuning of Haarnoja et al. 2018, arXiv:1812.05905):

  * Gaussian policy pi(u | s) = N(mu(s), sigma(s)) squashed by a = tanh(u), so
    that the action lies in [-1, 1]^3; the log-density includes the tanh
    correction  log pi(a|s) = log N(u) - sum log(1 - tanh(u)^2).
  * Twin critics Q1, Q2 with target copies updated by Polyak averaging
    (tau = 0.005); the target value uses the minimum of the two targets minus
    alpha * log pi (the entropy bonus).
  * The temperature alpha is learned so that the policy entropy stays near
    the target -|A| = -3 (loss  -log_alpha * (log pi + target_entropy)).
  * One gradient step of actor, critics and alpha per environment step,
    batches of 256 from a 200k ring buffer, Adam with lr 3e-4, gamma 0.99,
    MLPs of two hidden layers of 128 units (ReLU).

Parameter sharing. One actor and one pair of critics serve all nine units.
The unit identity is appended to the local observation as a 9-dim one-hot,
so the network input is 6 + 9 = 15 numbers, and every environment step
yields nine transitions (one per unit) with the same shared reward. The
policy is local at execution time: the action of unit i depends on its own
six observations and its own ID only. Sharing is used for sample efficiency:
one network sees nine views of the same feeder physics per step.

Day boundaries are time limits, not terminal states: the transition of the
last step of a day is stored with done = 0, so the critic bootstraps through
the boundary (the year is one continuing process; Pardo et al. 2018, "Time
limits in reinforcement learning", ICML).

Training reward and selection reward. The agent is trained on the reward of
env.VoltVarEnv.step, which includes the fence charge. The per-episode log
also carries the core reward (the same sum without the fence charge), and
`train_sac(best_key=...)` copies the weights (SACAgent.snapshot) after every
evaluation whose value of `best_key` is the best so far and returns them as
log["best"]; `save_snapshot` writes such a copy in the format of
SACAgent.save.

The module is plain torch (no RL library) and is deliberately small; the
training loop `train_sac` and the helpers `unit_obs`, `sac_policy` and
`sample_training_days` are what notebook 06 calls.
"""
import copy
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import env as envmod

N_UNITS = envmod.N_UNITS
OBS_DIM = envmod.OBS_DIM + N_UNITS          # 6 local + 9 one-hot = 15
ACT_DIM = envmod.ACT_DIM
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
_EYE = np.eye(N_UNITS, dtype=np.float32)


@dataclass
class SACConfig:
    hidden: int = 128
    n_layers: int = 2
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch: int = 256
    buffer: int = 200_000
    target_entropy: float = -float(ACT_DIM)
    alpha_init: float = 0.2
    n_threads: int = 2
    seed: int = 0

    def to_dict(self):
        return asdict(self)


def unit_obs(obs):
    """(9, 6) local observations -> (9, 15) with the unit one-hot appended."""
    return np.concatenate([np.asarray(obs, dtype=np.float32), _EYE], axis=1)


def _mlp(inp, out, hidden, n_layers):
    layers, d = [], inp
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = _mlp(OBS_DIM, 2 * ACT_DIM, cfg.hidden, cfg.n_layers)

    def forward(self, obs, deterministic=False, with_logp=True):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        if not with_logp:
            return a, None
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
        logp = logp - torch.log(1.0 - a.pow(2) + 1e-6).sum(-1, keepdim=True)     # tanh change of variables
        return a, logp


class Critic(nn.Module):
    """Twin Q networks on (s, a)."""
    def __init__(self, cfg):
        super().__init__()
        self.q1 = _mlp(OBS_DIM + ACT_DIM, 1, cfg.hidden, cfg.n_layers)
        self.q2 = _mlp(OBS_DIM + ACT_DIM, 1, cfg.hidden, cfg.n_layers)

    def forward(self, obs, act):
        sa = torch.cat([obs, act], dim=-1)
        return self.q1(sa), self.q2(sa)


class ReplayBuffer:
    def __init__(self, size, seed=0):
        self.size = size
        self.obs = np.zeros((size, OBS_DIM), np.float32); self.act = np.zeros((size, ACT_DIM), np.float32)
        self.rew = np.zeros(size, np.float32); self.nobs = np.zeros((size, OBS_DIM), np.float32)
        self.done = np.zeros(size, np.float32)
        self.ptr = 0; self.n = 0
        self.rng = np.random.default_rng(seed)

    def add_batch(self, obs, act, rew, nobs, done):
        """obs, nobs (k, 15), act (k, 3), scalar reward and done shared by the k rows."""
        k = len(obs)
        idx = (self.ptr + np.arange(k)) % self.size
        self.obs[idx] = obs; self.act[idx] = act; self.rew[idx] = rew; self.nobs[idx] = nobs; self.done[idx] = done
        self.ptr = (self.ptr + k) % self.size; self.n = min(self.n + k, self.size)

    def sample(self, batch):
        i = self.rng.integers(0, self.n, size=batch)
        f = torch.from_numpy
        return f(self.obs[i]), f(self.act[i]), f(self.rew[i]).unsqueeze(1), f(self.nobs[i]), f(self.done[i]).unsqueeze(1)


class SACAgent:
    def __init__(self, cfg=None):
        self.cfg = cfg or SACConfig()
        torch.set_num_threads(self.cfg.n_threads)
        torch.manual_seed(self.cfg.seed)
        self.actor = Actor(self.cfg); self.critic = Critic(self.cfg); self.critic_t = Critic(self.cfg)
        self.critic_t.load_state_dict(self.critic.state_dict())
        for p in self.critic_t.parameters():
            p.requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(self.cfg.alpha_init), dtype=torch.float32, requires_grad=True)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer, seed=self.cfg.seed)
        self.n_updates = 0

    @property
    def alpha(self):
        return float(self.log_alpha.exp())

    def act(self, obs15, deterministic=False):
        """(k, 15) numpy -> (k, 3) numpy action in [-1, 1]."""
        with torch.no_grad():
            a, _ = self.actor(torch.as_tensor(np.asarray(obs15, dtype=np.float32)), deterministic, with_logp=False)
        return a.numpy()

    def update(self):
        cfg = self.cfg
        obs, act, rew, nobs, done = self.buffer.sample(cfg.batch)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.actor(nobs)
            q1t, q2t = self.critic_t(nobs, a2)
            y = rew + cfg.gamma * (1.0 - done) * (torch.min(q1t, q2t) - alpha * logp2)
        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_critic.zero_grad(set_to_none=True); critic_loss.backward(); self.opt_critic.step()

        a, logp = self.actor(obs)
        q1p, q2p = self.critic(obs, a)
        actor_loss = (alpha * logp - torch.min(q1p, q2p)).mean()
        self.opt_actor.zero_grad(set_to_none=True); actor_loss.backward(); self.opt_actor.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + cfg.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True); alpha_loss.backward(); self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.mul_(1.0 - cfg.tau).add_(cfg.tau * p)
        self.n_updates += 1
        return dict(critic_loss=critic_loss.item(), actor_loss=actor_loss.item(), alpha=alpha.item(),
                    entropy=-logp.detach().mean().item(), q_mean=q1.detach().mean().item())

    def snapshot(self):
        """Deep copy of the weights in the format of save(), so that a
        checkpoint can be kept in memory while training goes on."""
        return dict(actor=copy.deepcopy(self.actor.state_dict()), critic=copy.deepcopy(self.critic.state_dict()),
                    log_alpha=self.log_alpha.detach().item(), config=self.cfg.to_dict(), n_updates=self.n_updates)

    def save(self, path):
        torch.save(self.snapshot(), path)

    def load(self, path):
        d = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(d["actor"]); self.critic.load_state_dict(d["critic"])
        self.critic_t.load_state_dict(d["critic"])
        with torch.no_grad():
            self.log_alpha.fill_(d["log_alpha"])
        self.n_updates = d.get("n_updates", 0)
        return self


def save_snapshot(snap, path):
    torch.save(snap, path)


def sac_policy(agent, deterministic=True):
    """policy_fn(obs, env) for env.run_policy: the nine units act on their
    own observation plus their one-hot ID through the shared actor."""
    return lambda obs, e: agent.act(unit_obs(obs), deterministic)


def sample_training_days(pv, n_days=120, seed=0, peak_threshold=0.5, weight_sunny=3.0):
    """n_days distinct days of the year, drawn without replacement with a
    weight of `weight_sunny` for days whose PV peak exceeds peak_threshold
    and 1 for the others. Returns (days sorted, peaks per day)."""
    irr = pv.values.reshape(365, 96, -1)
    peaks = irr.max(axis=(1, 2))
    w = np.where(peaks > peak_threshold, weight_sunny, 1.0); w = w / w.sum()
    rng = np.random.default_rng(seed)
    days = np.sort(rng.choice(365, size=n_days, replace=False, p=w))
    return days, peaks


def train_sac(E, agent, days, n_episodes, warmup=2000, eval_every=50, eval_fn=None, seed=0, verbose=True,
              log_every=100, best_key=None):
    """Train `agent` on environment E over `n_episodes` single-day episodes
    drawn (with replacement, in a seeded random order) from `days`. The
    first `warmup` environment steps use uniform random actions; from then
    on the stochastic policy acts and one gradient update follows every step.
    eval_fn(agent, episode) is called before the first episode and after
    every `eval_every` episodes (and at the end); it returns a dict that is
    stored in the log. If `best_key` names one of the keys of that dict, the
    weights are copied whenever the evaluation improves on the best value
    so far and returned as log["best"] = dict(episode, value, snapshot).
    Returns dict(episodes=dict of lists, evals=list of dicts, best=... or
    None, train_time_s, steps_total)."""
    rng = np.random.default_rng(seed)
    ep_log = dict(episode=[], day=[], reward=[], reward_core=[], fence_charge=[], alpha=[], critic_loss=[],
                  actor_loss=[], entropy=[], q_mean=[], safety_steps=[], safety_dq_kvarh=[], safety_dp_kwh=[],
                  hours_violation=[], steps_total=[], time_s=[])
    evals = []
    best = None

    def _record_eval(ep, step_total):
        nonlocal best
        evals.append(dict(episode=ep, steps=step_total, **eval_fn(agent, ep)))
        if best_key is not None and (best is None or evals[-1][best_key] > best["value"]):
            best = dict(episode=ep, value=float(evals[-1][best_key]), snapshot=agent.snapshot())

    step_total = 0
    t_start = time.perf_counter()
    if eval_fn is not None:
        _record_eval(0, 0)
    for ep in range(1, n_episodes + 1):
        day = int(rng.choice(days))
        obs = unit_obs(E.reset(day=day))
        t0 = time.perf_counter()
        ep_r = 0.0; ep_charge = 0.0; n_safe = 0; dq = 0.0; dp = 0.0; n_viol = 0
        losses = dict(critic_loss=[], actor_loss=[], alpha=[], entropy=[], q_mean=[])
        done = False
        while not done:
            if step_total < warmup:
                a = E.random_action().astype(np.float32)
            else:
                a = agent.act(obs, deterministic=False)
            nobs_raw, r, done, info = E.step(a)
            nobs = unit_obs(nobs_raw)
            agent.buffer.add_batch(obs, a, r, nobs, 0.0)         # time limit: bootstrap through the day boundary
            obs = nobs
            ep_r += r; ep_charge += info["terms"].get("safety", 0.0)
            n_safe += int(info["safety_active"]); dq += info["safety_dq"]; dp += info["safety_dp"]
            n_viol += int(info["excess_pu"] > 0)
            step_total += 1
            if step_total >= warmup and agent.buffer.n >= agent.cfg.batch:
                st = agent.update()
                for k in losses:
                    losses[k].append(st[k])
        ep_log["episode"].append(ep); ep_log["day"].append(day); ep_log["reward"].append(ep_r)
        ep_log["reward_core"].append(ep_r - ep_charge); ep_log["fence_charge"].append(ep_charge)
        for k in losses:
            ep_log[k].append(float(np.mean(losses[k])) if losses[k] else np.nan)
        ep_log["safety_steps"].append(n_safe); ep_log["safety_dq_kvarh"].append(dq * 0.25); ep_log["safety_dp_kwh"].append(dp * 0.25)
        ep_log["hours_violation"].append(n_viol * 0.25); ep_log["steps_total"].append(step_total)
        ep_log["time_s"].append(time.perf_counter() - t0)
        if verbose and (ep % log_every == 0 or ep == 1):
            print(f"  episode {ep:4d} day {day:3d}  reward {ep_r:8.2f}  alpha {ep_log['alpha'][-1]:.4f}  "
                  f"critic loss {ep_log['critic_loss'][-1]:8.4f}  fence {n_safe:2d} steps  "
                  f"{'(warm-up)' if step_total <= warmup else ''}  elapsed {time.perf_counter() - t_start:6.0f} s")
        if eval_fn is not None and (ep % eval_every == 0 or ep == n_episodes):
            _record_eval(ep, step_total)
    return dict(episodes=ep_log, evals=evals, best=best, train_time_s=time.perf_counter() - t_start, steps_total=step_total)
