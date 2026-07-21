"""tests/test_filter.py -- the five tests Nolan's particle filter core must pass.

These exercise filter.py's systematic_resample, pf_step (indirectly, via
run_filter), and run_filter. Every test below will raise NotImplementedError
until filter.py's stubs are filled in -- that's expected; this file exists so
the four verbs (predict/weight/normalize/resample) can be built and checked
incrementally.

Run with: pytest -m "not slow"   (test_coverage is the one slow test, ~20 filter
runs; run the full suite with plain `pytest` once the fast tests are green.)
"""

import numpy as np
import pytest

from config import Config, logit
from filter import run_filter, systematic_resample
from simulator import simulate, simulate_observations

ALL_CHANNELS = {
    "audits": True, "depots": True, "fires": True, "prices": True,
    "mailbox": True, "compactor": True, "tags": True,
}


def test_resample_preserves_mean():
    """Resampling a known weighted cloud should approximately preserve its weighted mean."""
    rng = np.random.default_rng(42)
    n = 4000
    values = rng.normal(loc=5.0, scale=2.0, size=n)
    # Skew the weights toward one tail so the weighted mean differs clearly from the raw mean.
    raw_weights = np.exp(-0.5 * ((values - 6.5) / 1.5) ** 2)
    weights = raw_weights / raw_weights.sum()
    weighted_mean_before = np.sum(values * weights)

    idx = systematic_resample(weights, rng)

    assert idx.shape == (n,)
    assert idx.dtype.kind in ("i", "u")
    assert idx.min() >= 0 and idx.max() < n

    resampled_mean = values[idx].mean()
    assert abs(resampled_mean - weighted_mean_before) < 0.15


def test_tracks_constant_truth():
    """With truth held constant and every channel on, posterior median bias on phi_bar is small."""
    cfg = Config()
    rng = np.random.default_rng(123)
    T = cfg.geography.n_quarters
    phi_bar_true = 0.30
    w_emb, w_rem = cfg.geography.class_share
    x_const = 250.0

    truth = {
        "X": np.full(T, x_const),
        "X_emb": np.full(T, w_emb * x_const),
        "X_rem": np.full(T, w_rem * x_const),
        "phi_emb": np.full(T, phi_bar_true),
        "phi_rem": np.full(T, phi_bar_true),
        "phi_bar": np.full(T, phi_bar_true),
        "r": np.zeros(T, dtype=int),
        "s": np.full(T, cfg.price_spread.mu[0]),
        "L": np.full(T, cfg.fire_intensity.l0),
        "u": np.zeros(T),
    }
    cfg.tag.tau = 0.2
    obs = simulate_observations(cfg, rng, truth)

    results = run_filter(obs, ALL_CHANNELS, cfg, n_particles=3000, seed=7)
    median_phi_bar = results["quantiles"]["phi_bar"][:, 1]

    # Give the filter a few quarters to burn in (mailbox/tags take time to become
    # informative), then check bias over the back half of the horizon.
    bias = np.abs(median_phi_bar[20:] - phi_bar_true)
    assert bias.mean() < 0.04


def test_regime_detected():
    """Posterior mean of s_t moves toward the new regime mean within 3 quarters of t_shock."""
    cfg = Config()
    cfg.phi_dynamics.t_deposit = None
    cfg.tag.tau = 0.0
    rng = np.random.default_rng(5)
    scenario = simulate(cfg, rng)
    obs = scenario["obs"]
    active_channels = {
        "audits": True, "depots": True, "fires": True, "prices": True,
        "mailbox": False, "compactor": False, "tags": False,
    }

    results = run_filter(obs, active_channels, cfg, n_particles=3000, seed=11)
    particles_s = results["particles_history"]["s"]  # (T, N)
    weights = results["weights_history"]
    posterior_mean_s = np.sum(particles_s * weights, axis=1)

    t_shock = cfg.regime.t_shock
    old_mean, new_mean = cfg.price_spread.mu[0], cfg.price_spread.mu[1]
    gap = abs(new_mean - old_mean)

    window = posterior_mean_s[t_shock: t_shock + 3]
    moved_fraction = np.abs(window - old_mean) / gap
    assert np.max(moved_fraction) > 0.5


def test_more_channels_narrower():
    """Mean 90% band width on phi_bar strictly narrows, averaged over 5 seeds,
    going audits-only -> +mailbox -> +compactor -> +tags(0.2)."""
    cfg = Config()
    seeds = range(5)
    stages = [
        {"audits": True, "depots": False, "fires": False, "prices": False,
         "mailbox": False, "compactor": False, "tags": False},
        {"audits": True, "depots": False, "fires": False, "prices": False,
         "mailbox": True, "compactor": False, "tags": False},
        {"audits": True, "depots": False, "fires": False, "prices": False,
         "mailbox": True, "compactor": True, "tags": False},
        {"audits": True, "depots": False, "fires": False, "prices": False,
         "mailbox": True, "compactor": True, "tags": True},
    ]
    tau = 0.2

    cfg.phi_dynamics.t_deposit = None
    mean_widths = []
    for stage in stages:
        cfg.tag.tau = tau if stage["tags"] else 0.0
        widths = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            scenario = simulate(cfg, rng)
            results = run_filter(scenario["obs"], stage, cfg, n_particles=3000, seed=seed)
            q = results["quantiles"]["phi_bar"]
            widths.append((q[:, 2] - q[:, 0]).mean())
        mean_widths.append(np.mean(widths))

    assert mean_widths[0] > mean_widths[1] > mean_widths[2] > mean_widths[3]


@pytest.mark.slow
def test_coverage():
    """Over 20 seeds, the 90% band covers true phi_bar in roughly 85-95% of quarter-instances."""
    cfg = Config()
    active_channels = {
        "audits": True, "depots": True, "fires": True, "prices": True,
        "mailbox": True, "compactor": True, "tags": False,
    }
    cfg.phi_dynamics.t_deposit = 20
    cfg.tag.tau = 0.0
    covered, total = 0, 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        scenario = simulate(cfg, rng)
        truth, obs = scenario["truth"], scenario["obs"]
        results = run_filter(obs, active_channels, cfg, n_particles=3000, seed=seed)
        q = results["quantiles"]["phi_bar"]
        lo, hi = q[:, 0], q[:, 2]
        covered += int(np.sum((truth["phi_bar"] >= lo) & (truth["phi_bar"] <= hi)))
        total += cfg.geography.n_quarters

    coverage = covered / total
    assert 0.80 <= coverage <= 0.97
