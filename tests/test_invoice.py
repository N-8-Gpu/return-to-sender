"""tests/test_invoice.py -- guards on the cost math behind the invoice.

These exist because the one real bug caught during development (u_sort set
above both regime price means, structurally zeroing the destroyed-material
line item) was only found by eyeballing a rendered figure. These make that
class of bug, and the quantile bookkeeping around it, fail loudly instead.
"""

import numpy as np

from config import Config
from invoice import (
    compute_cost_posterior,
    modulation_table,
    per_particle_cost_items,
    weighted_quantile,
)
from simulator import simulate


def _fake_results(cfg: Config, n: int = 400, seed: int = 0) -> dict:
    """A truth-hugging particle cloud with uniform weights, standing in for run_filter output."""
    rng = np.random.default_rng(seed)
    scenario = simulate(cfg, rng)
    truth = scenario["truth"]
    T = cfg.geography.n_quarters
    return {
        "particles_history": {
            "phi_bar": np.clip(truth["phi_bar"][:, None] + rng.normal(0, 0.02, (T, n)), 1e-3, 1 - 1e-3),
            "X": truth["X"][:, None] * rng.lognormal(0, 0.05, (T, n)),
            "L": truth["L"][:, None] + rng.normal(0, 0.05, (T, n)),
            "s": truth["s"][:, None] + rng.normal(0, 5, (T, n)),
        },
        "weights_history": np.full((T, n), 1.0 / n),
    }


def test_line_items_sum_to_total():
    """The three line items must sum to the total, particle by particle."""
    cfg = Config()
    rng = np.random.default_rng(1)
    phi_bar = rng.uniform(0.05, 0.6, (10, 50))
    X = rng.uniform(50, 400, (10, 50))
    L = rng.normal(-1.7, 0.3, (10, 50))
    s = rng.normal(120, 40, (10, 50))
    total, landfill, fire, material = per_particle_cost_items(phi_bar, X, L, s, cfg)
    np.testing.assert_allclose(total, landfill + fire + material, rtol=1e-12)
    assert np.all(landfill >= 0) and np.all(fire >= 0) and np.all(material >= 0)


def test_material_item_not_structurally_zero():
    """u_sort must sit below achievable price spreads, or the material line item can never bite."""
    cfg = Config()
    assert cfg.invoice.u_sort < max(cfg.price_spread.mu), (
        "u_sort is at or above every regime's mean price spread; "
        "the destroyed-material line item would be structurally zero"
    )
    posterior = compute_cost_posterior(_fake_results(cfg), cfg)
    assert posterior["material_total"] > 0


def test_cumulative_cost_monotone_and_ordered():
    """Cumulative-cost traces never decrease, and the floor/median/p95 ordering holds."""
    cfg = Config()
    posterior = compute_cost_posterior(_fake_results(cfg), cfg)
    assert np.all(np.diff(posterior["cumulative_median"]) >= -1e-9)
    assert np.all(np.diff(posterior["cumulative_p05"]) >= -1e-9)
    assert posterior["billable_floor"] <= posterior["total_median"] <= posterior["total_p95"]


def test_modulation_recovers_base_fee():
    """Class-share-weighted average of the modulated rates must equal the aggregate per-tonne fee."""
    cfg = Config()
    mod = modulation_table(total_cost=1_000_000.0, total_leaked_tonnage=250.0, cfg=cfg)
    w_emb, w_rem = cfg.geography.class_share
    weighted_avg = w_emb * mod["embedded_rate"] + w_rem * mod["removable_rate"]
    np.testing.assert_allclose(weighted_avg, 1_000_000.0 / 250.0, rtol=1e-12)
    np.testing.assert_allclose(mod["embedded_rate"] / mod["removable_rate"], mod["ratio"], rtol=1e-12)


def test_weighted_quantile_matches_numpy_under_uniform_weights():
    """With uniform weights, the weighted quantile should track numpy's within one grid step."""
    rng = np.random.default_rng(3)
    values = rng.normal(0, 1, 5000)
    weights = np.full(5000, 1 / 5000)
    for q in (0.05, 0.5, 0.95):
        ours = weighted_quantile(values, weights, q)
        numpys = float(np.quantile(values, q))
        assert abs(ours - numpys) < 0.02
