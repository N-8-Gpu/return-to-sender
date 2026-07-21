"""invoice.py -- cost computation and the rendered "externality invoice" figure.

Cost model (brief): C_t = phi_bar_t * X_t * (u_LF + exp(L_t)*u_F + max(s_t - u_sort, 0)),
decomposed into three line items: landfill space & perpetual care, expected fire
cost, and destroyed material value. Given the filter's particle cloud, this module
turns per-particle per-quarter cost into a posterior over CUMULATIVE cost (the
running sum of C_t so far): a median trace, a 90% band, and a conservative
5th-percentile "billable floor" -- then renders that as a styled invoice figure.

Contract with filter.py's run_filter(...) results dict (see filter.py docstring):
  results['particles_history'][state]   # array (T, N), state in {'X','phi_bar','L','s'}
  results['weights_history']            # array (T, N), normalized weights per quarter
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from config import Config


def per_particle_cost_items(
    phi_bar: np.ndarray, X: np.ndarray, L: np.ndarray, s: np.ndarray, cfg: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-quarter, per-particle cost and its three line items.

    Inputs share one broadcastable shape -- a single quarter's (N,) particle
    slice or a full (T, N) history both work. Returns (total, landfill, fire,
    material), each the same shape as the inputs.
    """
    leaked = phi_bar * X
    landfill = leaked * cfg.invoice.u_lf
    fire = leaked * np.exp(L) * cfg.invoice.u_f
    material = leaked * np.maximum(s - cfg.invoice.u_sort, 0.0)
    total = landfill + fire + material
    return total, landfill, fire, material


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile of a 1-D array of particle values (weights sum to ~1)."""
    order = np.argsort(values)
    v_sorted = values[order]
    cum_w = np.cumsum(weights[order])
    cum_w /= cum_w[-1]  # guard against floating-point drift away from exactly 1
    idx = min(int(np.searchsorted(cum_w, q)), len(v_sorted) - 1)
    return float(v_sorted[idx])


def weighted_quantile_trace(history: np.ndarray, weights_history: np.ndarray, q: float) -> np.ndarray:
    """Weighted quantile at each quarter, for a (T, N) particle history."""
    T = history.shape[0]
    return np.array([weighted_quantile(history[t], weights_history[t], q) for t in range(T)])


def compute_cost_posterior(results: dict, cfg: Config) -> dict:
    """Turn the filter's particle history into the invoice's cost posterior.

    Returns per-quarter median/90%-band CUMULATIVE cost, the final billable
    floor, posterior-median totals for each of the three line items over the
    full horizon, and the posterior-median cumulative leaked tonnage (for the
    modulation table's per-tonne rate split).
    """
    ph = results["particles_history"]
    wh = results["weights_history"]
    phi_bar, X, L, s = ph["phi_bar"], ph["X"], ph["L"], ph["s"]

    total, landfill, fire, material = per_particle_cost_items(phi_bar, X, L, s, cfg)
    cum_total = np.cumsum(total, axis=0)
    cum_landfill = np.cumsum(landfill, axis=0)
    cum_fire = np.cumsum(fire, axis=0)
    cum_material = np.cumsum(material, axis=0)
    cum_leaked = np.cumsum(phi_bar * X, axis=0)

    median_trace = weighted_quantile_trace(cum_total, wh, 0.5)
    lo_trace = weighted_quantile_trace(cum_total, wh, 0.05)
    hi_trace = weighted_quantile_trace(cum_total, wh, 0.95)

    return {
        "cumulative_median": median_trace,
        "cumulative_p05": lo_trace,
        "cumulative_p95": hi_trace,
        "billable_floor": lo_trace[-1],
        "total_median": float(median_trace[-1]),
        "landfill_total": weighted_quantile(cum_landfill[-1], wh[-1], 0.5),
        "fire_total": weighted_quantile(cum_fire[-1], wh[-1], 0.5),
        "material_total": weighted_quantile(cum_material[-1], wh[-1], 0.5),
        "total_leaked_tonnage": weighted_quantile(cum_leaked[-1], wh[-1], 0.5),
    }


def modulation_table(total_cost: float, total_leaked_tonnage: float, cfg: Config) -> dict:
    """Split the aggregate per-tonne stewardship fee into an embedded and a
    removable rate, in the ratio fixed by cfg.invoice.modulation_ratio, such
    that the class-share-weighted average recovers the aggregate fee.
    """
    w_emb, w_rem = cfg.geography.class_share
    ratio = cfg.invoice.modulation_ratio
    base_fee = total_cost / total_leaked_tonnage if total_leaked_tonnage > 0 else 0.0
    removable_rate = base_fee / (w_emb * ratio + w_rem)
    embedded_rate = ratio * removable_rate
    return {"embedded_rate": embedded_rate, "removable_rate": removable_rate, "ratio": ratio}


def render_invoice(posterior: dict, modulation: dict, seed: int) -> plt.Figure:
    """Render the externality invoice: three line items, the billable floor
    highlighted, and a modulation table -- styled to look like a bill, not a chart.
    """
    fig, ax = plt.subplots(figsize=(6.5, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(mpatches.Rectangle((0.03, 0.02), 0.94, 0.96, fill=False, linewidth=1.5, edgecolor="black"))

    ax.text(0.5, 0.94, "EXTERNALITY INVOICE", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(
        0.5, 0.905,
        "Return to Sender -- battery-waste cost accounting (synthetic demo)",
        ha="center", va="center", fontsize=9, style="italic", color="dimgray",
    )

    ax.text(0.07, 0.85, "BILL TO:", fontsize=9, fontweight="bold")
    ax.text(0.07, 0.825, "Consumer Lithium-Ion Battery Producers,", fontsize=10)
    ax.text(0.07, 0.80, "via stewardship organization", fontsize=10)
    ax.text(0.75, 0.85, "SEED", fontsize=9, fontweight="bold")
    ax.text(0.75, 0.825, str(seed), fontsize=10)

    ax.plot([0.07, 0.93], [0.77, 0.77], color="black", linewidth=1)

    line_items = [
        ("Landfill space & perpetual care", posterior["landfill_total"]),
        ("Expected fire cost", posterior["fire_total"]),
        ("Destroyed material value", posterior["material_total"]),
    ]
    y = 0.72
    for label, amount in line_items:
        ax.text(0.07, y, label, fontsize=11)
        ax.text(0.93, y, f"${amount:,.0f}", fontsize=11, ha="right")
        y -= 0.045

    y += 0.02
    ax.plot([0.07, 0.93], [y, y], color="black", linewidth=0.8)
    y -= 0.045
    ax.text(0.07, y, "TOTAL (posterior median)", fontsize=12, fontweight="bold")
    ax.text(0.93, y, f"${posterior['total_median']:,.0f}", fontsize=12, fontweight="bold", ha="right")

    y -= 0.09
    box_h = 0.06
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.07, y - box_h), 0.86, box_h, boxstyle="round,pad=0.01",
        facecolor="#fde8e8", edgecolor="firebrick",
    ))
    ax.text(
        0.5, y - box_h / 2,
        f"Billable floor (5th percentile, conservative): ${posterior['billable_floor']:,.0f}",
        fontsize=10.5, ha="center", va="center", color="firebrick", fontweight="bold",
    )

    y -= box_h + 0.09
    ax.text(0.07, y, "MODULATION BY DESIGN CLASS", fontsize=10, fontweight="bold")
    y -= 0.04
    ax.text(0.10, y, "Embedded (non-removable) rate", fontsize=10)
    ax.text(0.93, y, f"${modulation['embedded_rate']:,.2f} / tonne", fontsize=10, ha="right")
    y -= 0.035
    ax.text(0.10, y, "Removable rate", fontsize=10)
    ax.text(0.93, y, f"${modulation['removable_rate']:,.2f} / tonne", fontsize=10, ha="right")
    y -= 0.035
    ax.text(0.10, y, "Ratio (embedded : removable)", fontsize=10)
    ax.text(0.93, y, f"{modulation['ratio']:.2f} : 1", fontsize=10, ha="right")

    ax.text(
        0.5, 0.05, "Synthetic data, illustrative model -- not a real invoice.",
        fontsize=8, ha="center", color="dimgray", style="italic",
    )

    fig.tight_layout()
    return fig
