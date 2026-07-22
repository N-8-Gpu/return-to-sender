"""filter.py -- the particle filter core.

Fills in the four verbs:

    for each quarter t:
        GUESS:  push every particle through the dynamics (sample noise, flip regime coin)
        WEIGHT: w_i *= product over active channels of likelihood(obs_t | particle_i)
        NORM:   w /= sum(w);  N_eff = 1 / sum(w**2)
        CULL:   if N_eff < N/2: systematic resample, reset w to 1/N
        RECORD: weighted quantiles of each state

Same predict-correct heartbeat as a Kalman filter -- sampling replaces algebra.

particles: a dict of arrays, one per hidden state, each shape (n_particles,).
Keys: 'log_x', 'logit_phi_emb', 'logit_phi_rem', 'r', 's', 'L'. phi_emb/
phi_rem/phi_bar/X are derived from these via config.sigmoid / np.exp -- the
particle cloud stays in the same internal representation the dynamics use,
converted to natural units only when recording results.

active_channels: dict mapping channel name -> bool, e.g.
{'audits': True, 'depots': True, 'fires': False, 'prices': True,
 'mailbox': True, 'compactor': True, 'tags': False}.

Three quantities in the GUESS/WEIGHT steps depend on the quarter t, not on
any particle: predict_x's anchor_t (the kappa-weighted sales history),
mailbox_loglik's uptake_t (the mailback ramp fraction), and predict_phi_logit's
u_t (the deposit-intervention indicator, 0 before params.phi_dynamics.t_deposit
and 1 from then on -- None means it never switches on). pf_step's signature
has no `t` argument, and predict_phi_logit is called from inside pf_step, so
all three ride along in obs_t under the keys '_anchor', '_mailbox_uptake',
and '_u', injected by run_filter before each pf_step call. tag_loglik reads
params.tag.tau directly instead (a scenario-level constant, not quarter-
varying, so it doesn't need routing through obs_t).
"""

from __future__ import annotations

import numpy as np

from config import Config, logit, sigmoid
from simulator import (
    audit_loglik,
    compactor_loglik,
    depot_loglik,
    fire_loglik,
    mailbox_loglik,
    mailbox_uptake,
    make_x_anchor_series,
    predict_fire_intensity,
    predict_phi_logit,
    predict_price_spread,
    predict_regime,
    predict_x,
    price_loglik,
    tag_loglik,
)

_PARTICLE_KEYS = ("log_x", "logit_phi_emb", "logit_phi_rem", "r", "s", "L")
_QUANTILE_STATES = ("X", "phi_emb", "phi_rem", "phi_bar", "s", "L")
_HISTORY_STATES = ("X", "phi_bar", "L", "s")
_QUANTILE_LEVELS = (0.05, 0.5, 0.95)


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling: draw n_particles indices with probability
    proportional to `weights` (which must sum to 1), using one random offset
    and evenly spaced strata rather than n_particles independent draws (lower
    variance than naive multinomial resampling).

    Returns an integer array of shape (len(weights),): the index into the
    original particle arrays that each new particle should copy.
    """
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard against floating-point drift below 1
    return np.searchsorted(cumulative, positions)


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The (5th, 50th, 95th) weighted percentile of a 1-D particle array."""
    n = len(values)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    idx = np.searchsorted(cumulative, _QUANTILE_LEVELS)
    idx = np.minimum(idx, n - 1)
    return values[order][idx]


def pf_step(
    particles: dict[str, np.ndarray],
    weights: np.ndarray,
    obs_t: dict[str, float],
    active_channels: dict[str, bool],
    params: Config,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Advance the particle cloud by one quarter: GUESS, WEIGHT, NORM, CULL.

    particles: current particle cloud (see module docstring for keys/shapes).
    weights: current normalized weights, shape (n_particles,), summing to 1.
    obs_t: this quarter's observations, one scalar per channel (may be NaN),
    plus the quarter-dependent constants '_anchor', '_mailbox_uptake', '_u'.
    active_channels: which channels the filter should condition on this run.
    params: the shared Config instance (dynamics + observation-noise settings).
    rng: shared numpy Generator, already seeded by the caller.

    Returns (new_particles, new_weights, n_eff) where n_eff = 1 / sum(weights**2)
    is computed AFTER normalizing but BEFORE any resampling.
    """
    n = len(weights)
    anchor_t = obs_t["_anchor"]
    uptake_t = obs_t["_mailbox_uptake"]
    u_t = obs_t["_u"]

    # GUESS
    log_x = predict_x(particles["log_x"], anchor_t, params, rng)
    logit_phi_emb = predict_phi_logit(particles["logit_phi_emb"], u_t, params, rng)
    logit_phi_rem = predict_phi_logit(particles["logit_phi_rem"], u_t, params, rng)
    r = predict_regime(particles["r"], params, rng)
    s = predict_price_spread(particles["s"], r, params, rng)
    L = predict_fire_intensity(particles["L"], params, rng)

    phi_emb = sigmoid(logit_phi_emb)
    phi_rem = sigmoid(logit_phi_rem)
    w_emb, w_rem = params.geography.class_share
    phi_bar = w_emb * phi_emb + w_rem * phi_rem
    X = np.exp(log_x)
    X_emb, X_rem = w_emb * X, w_rem * X

    # WEIGHT: sum log-likelihoods over active channels, then exponentiate
    log_w = np.log(weights)
    if active_channels.get("audits"):
        log_w = log_w + audit_loglik(obs_t["audits_emb"], phi_emb, params)
        log_w = log_w + audit_loglik(obs_t["audits_rem"], phi_rem, params)
    if active_channels.get("depots"):
        log_w = log_w + depot_loglik(obs_t["depots"], phi_bar, X, params)
    if active_channels.get("fires"):
        log_w = log_w + fire_loglik(obs_t["fires"], phi_emb, phi_rem, X_emb, X_rem, L, params)
    if active_channels.get("prices"):
        log_w = log_w + price_loglik(obs_t["prices"], s, params)
    if active_channels.get("mailbox"):
        log_w = log_w + mailbox_loglik(obs_t["mailbox"], phi_bar, X, uptake_t, params)
    if active_channels.get("compactor"):
        log_w = log_w + compactor_loglik(obs_t["compactor"], phi_bar, X, params)
    if active_channels.get("tags"):
        log_w = log_w + tag_loglik(obs_t["tags"], phi_bar, params)

    # NORM
    log_w -= log_w.max()  # stability: keeps exp() from overflowing/underflowing
    new_weights = np.exp(log_w)
    new_weights /= new_weights.sum()
    n_eff = 1.0 / np.sum(new_weights ** 2)

    new_particles = {
        "log_x": log_x, "logit_phi_emb": logit_phi_emb, "logit_phi_rem": logit_phi_rem,
        "r": r, "s": s, "L": L,
    }

    # CULL
    if n_eff < n / 2:
        idx = systematic_resample(new_weights, rng)
        new_particles = {key: value[idx] for key, value in new_particles.items()}
        new_weights = np.full(n, 1.0 / n)

    return new_particles, new_weights, n_eff


def run_filter(
    observations: dict[str, np.ndarray],
    active_channels: dict[str, bool],
    params: Config,
    n_particles: int = 5000,
    seed: int = 0,
) -> dict:
    """Run the particle filter over every quarter in `observations`.

    observations: dict keyed by channel name -> array of shape (T,), as
    returned by simulator.simulate(...)['obs']. Includes channels that are
    off in active_channels; pf_step ignores those via the active_channels check.

    Initializes the particle cloud by sampling n_particles draws from the same
    priors simulate_truth uses for quarter -1 (params.phi_dynamics.phi0,
    params.fire_intensity.l0, the lifespan anchor, etc.) plus one step of
    process noise, then calls pf_step once per quarter.

    Reads params.phi_dynamics.t_deposit and params.tag.tau to replicate the
    same deposit-intervention timing and tag-adoption share used to generate
    `observations` -- set both on `params` to match before calling this.

    Returns a dict with:
      - 'quantiles': dict mapping state name -> array of shape (T, 3), the
        weighted (5th, 50th, 95th) percentile at each quarter, for 'X',
        'phi_emb', 'phi_rem', 'phi_bar', 's', 'L'.
      - 'n_eff': array of shape (T,), the N_eff computed each quarter.
      - 'particles_history': dict mapping state name -> array of shape
        (T, n_particles), for 'X', 'phi_bar', 'L', 's' -- the particle values
        used to compute that quarter's weights, before any resampling.
      - 'weights_history': array of shape (T, n_particles), the normalized
        weights paired with 'particles_history' at each quarter.
    """
    rng = np.random.default_rng(seed)
    T = params.geography.n_quarters
    n = n_particles

    anchors = make_x_anchor_series(params)
    w_emb, w_rem = params.geography.class_share
    t_deposit = params.phi_dynamics.t_deposit

    logit_phi_emb0, logit_phi_rem0 = logit(np.array(params.phi_dynamics.phi0))
    particles = {
        "log_x": np.full(n, anchors[0]) + rng.normal(0.0, np.sqrt(params.x_dynamics.q_x), n),
        "logit_phi_emb": np.full(n, logit_phi_emb0) + rng.normal(0.0, np.sqrt(params.phi_dynamics.q_phi), n),
        "logit_phi_rem": np.full(n, logit_phi_rem0) + rng.normal(0.0, np.sqrt(params.phi_dynamics.q_phi), n),
        "r": np.zeros(n, dtype=int),
        "s": np.full(n, params.price_spread.mu[0]),
        "L": np.full(n, params.fire_intensity.l0) + rng.normal(0.0, np.sqrt(params.fire_intensity.q_l), n),
    }
    weights = np.full(n, 1.0 / n)

    quantiles = {name: np.empty((T, 3)) for name in _QUANTILE_STATES}
    n_eff_trace = np.empty(T)
    particles_history = {name: np.empty((T, n)) for name in _HISTORY_STATES}
    weights_history = np.empty((T, n))

    obs_channels = ("audits_emb", "audits_rem", "depots", "fires", "prices", "mailbox", "compactor", "tags")

    for t in range(T):
        obs_t = {ch: observations[ch][t] if ch in observations else np.nan for ch in obs_channels}
        obs_t["_anchor"] = anchors[t]
        obs_t["_mailbox_uptake"] = mailbox_uptake(t, params)
        obs_t["_u"] = 1.0 if (t_deposit is not None and t >= t_deposit) else 0.0

        particles, weights, n_eff = pf_step(particles, weights, obs_t, active_channels, params, rng)
        n_eff_trace[t] = n_eff

        phi_emb = sigmoid(particles["logit_phi_emb"])
        phi_rem = sigmoid(particles["logit_phi_rem"])
        phi_bar = w_emb * phi_emb + w_rem * phi_rem
        X = np.exp(particles["log_x"])

        state_values = {
            "X": X, "phi_emb": phi_emb, "phi_rem": phi_rem, "phi_bar": phi_bar,
            "s": particles["s"], "L": particles["L"],
        }
        for name in _QUANTILE_STATES:
            quantiles[name][t] = _weighted_quantiles(state_values[name], weights)
        for name in _HISTORY_STATES:
            particles_history[name][t] = state_values[name]
        weights_history[t] = weights

    return {
        "quantiles": quantiles,
        "n_eff": n_eff_trace,
        "particles_history": particles_history,
        "weights_history": weights_history,
    }
