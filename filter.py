"""filter.py -- the particle filter core. HUMAN-WRITTEN (Nolan's part).

This file intentionally contains only signatures, docstrings, and
NotImplementedError stubs. Fill in the four verbs below:

    for each quarter t:
        GUESS:  push every particle through the dynamics (sample noise, flip regime coin)
        WEIGHT: w_i *= product over active channels of likelihood(obs_t | particle_i)
        NORM:   w /= sum(w);  N_eff = 1 / sum(w**2)
        CULL:   if N_eff < N/2: systematic resample, reset w to 1/N
        RECORD: weighted quantiles of each state

Same predict-correct heartbeat as a Kalman filter -- sampling replaces algebra.

Dynamics: call simulator.py's predict_x, predict_phi_logit, predict_regime,
predict_price_spread, predict_fire_intensity for the GUESS step (they're
already vectorized over a particle array, one call per quarter).

Weighting: call simulator.py's audit_loglik, depot_loglik, fire_loglik,
price_loglik, mailbox_loglik, compactor_loglik, tag_loglik for the WEIGHT
step. Each returns a per-particle log-likelihood array; sum the logs across
active channels, then exponentiate (subtract the max first for stability)
and multiply into the weights. Every one of these already returns zeros for
a NaN observation, so it's safe to call them even for channels that are
toggled off or not yet active -- but check `active_channels` too, since a
channel the user has switched off should not be conditioned on even when
data exists.

particles: a dict of arrays, one per hidden state, each shape (n_particles,).
Keys: 'log_x', 'logit_phi_emb', 'logit_phi_rem', 'r', 's', 'L'.
(phi_emb/phi_rem/phi_bar/X are derived from these via config.sigmoid /
np.exp -- keep the particle cloud in the same internal representation the
dynamics use, convert to natural units only when recording results.)

active_channels: dict mapping channel name -> bool, e.g.
{'audits': True, 'depots': True, 'fires': False, 'prices': True,
 'mailbox': True, 'compactor': True, 'tags': False}.
"""

from __future__ import annotations

import numpy as np

from config import Config


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling: draw n_particles indices with probability
    proportional to `weights` (which must sum to 1), using one random offset
    and evenly spaced strata rather than n_particles independent draws (lower
    variance than naive multinomial resampling).

    Returns an integer array of shape (len(weights),): the index into the
    original particle arrays that each new particle should copy.
    """
    raise NotImplementedError("Nolan's part: implement systematic resampling.")


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
    obs_t: this quarter's observations, one scalar per channel (may be NaN).
    active_channels: which channels the filter should condition on this run.
    params: the shared Config instance (dynamics + observation-noise settings).
    rng: shared numpy Generator, already seeded by the caller.

    Returns (new_particles, new_weights, n_eff) where n_eff = 1 / sum(weights**2)
    is computed AFTER normalizing but BEFORE any resampling (it's the
    diagnostic that decides whether to resample, and the value the caller
    records for the N_eff sparkline).
    """
    raise NotImplementedError("Nolan's part: implement predict/weight/normalize/resample.")


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
    off in active_channels (pf_step is responsible for ignoring those).

    Initializes the particle cloud by sampling n_particles draws from the
    same priors simulate_truth uses for quarter -1 (params.phi_dynamics.phi0,
    params.fire_intensity.l0, the lifespan anchor, etc.) plus process noise
    for one step, then calls pf_step once per quarter.

    Returns a dict holding, at minimum:
      - per-quarter weighted quantiles (e.g. 5th/50th/95th percentile) for
        each of X, phi_emb, phi_rem, phi_bar, r, s, L
      - the N_eff trace, one value per quarter
      - the final particle cloud and weights (invoice.py needs these to
        compute the cost posterior)
    """
    raise NotImplementedError("Nolan's part: implement the per-quarter filter loop.")
