"""invoice.py -- cost computation and the rendered "externality invoice" figure.

Cost model (brief): C_t = phi_bar_t * X_t * (u_LF + exp(L_t)*u_F + max(s_t - u_sort, 0)),
decomposed into three line items: landfill space & perpetual care, expected fire
cost, and destroyed material value. Given the filter's particle cloud, this module
turns per-particle per-quarter cost into a posterior over CUMULATIVE cost (the
running sum of C_t so far): a median trace, a 90% band, and a conservative
5th-percentile "billable floor" -- then renders that as a styled invoice figure.

The billable floor is the legally load-bearing number: it is the amount that
remains defensible even under conservative assumptions, so it is presented as
the payable amount. Line items are posterior medians and medians are not
additive -- the rendered invoice says so explicitly rather than pretending the
column sums.

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
    full horizon, the posterior-median cumulative leaked tonnage (for the
    modulation table's per-tonne rate split), and the posterior-median expected
    fire-incident count (the fire line item's quantity column).
    """
    ph = results["particles_history"]
    wh = results["weights_history"]
    phi_bar, X, L, s = ph["phi_bar"], ph["X"], ph["L"], ph["s"]

    total, landfill, fire, material = per_particle_cost_items(phi_bar, X, L, s, cfg)
    cum_total = np.cumsum(total, axis=0)
    cum_landfill = np.cumsum(landfill, axis=0)
    cum_fire = np.cumsum(fire, axis=0)
    cum_material = np.cumsum(material, axis=0)
    leaked = phi_bar * X
    cum_leaked = np.cumsum(leaked, axis=0)
    cum_expected_fires = np.cumsum(leaked * np.exp(L), axis=0)  # invoice-basis fire rate

    median_trace = weighted_quantile_trace(cum_total, wh, 0.5)
    lo_trace = weighted_quantile_trace(cum_total, wh, 0.05)
    hi_trace = weighted_quantile_trace(cum_total, wh, 0.95)

    return {
        "cumulative_median": median_trace,
        "cumulative_p05": lo_trace,
        "cumulative_p95": hi_trace,
        "billable_floor": lo_trace[-1],
        "total_median": float(median_trace[-1]),
        "total_p95": float(hi_trace[-1]),
        "landfill_total": weighted_quantile(cum_landfill[-1], wh[-1], 0.5),
        "fire_total": weighted_quantile(cum_fire[-1], wh[-1], 0.5),
        "material_total": weighted_quantile(cum_material[-1], wh[-1], 0.5),
        "total_leaked_tonnage": weighted_quantile(cum_leaked[-1], wh[-1], 0.5),
        "expected_fires_total": weighted_quantile(cum_expected_fires[-1], wh[-1], 0.5),
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


def _money(x: float) -> str:
    # \$ keeps matplotlib from reading paired dollar signs in one string as mathtext
    return rf"\${x:,.0f}"


def render_invoice(posterior: dict, modulation: dict, seed: int, meta: dict | None = None) -> plt.Figure:
    """Render the externality estimate as a consulting-grade illustrative assessment.

    meta (all optional, sensible defaults):
      'ref'      -- invoice reference string, e.g. 'RTS-2026-000'
      'period'   -- billing period label, e.g. 'Quarters 1-40 (10-year horizon)'
      'basis'    -- data basis label, e.g. 'Demonstration scenario (synthetic data)'
      'channels' -- list of active evidence-channel names cited as the basis
    """
    meta = meta or {}
    ref = meta.get("ref", f"RTS-2026-{seed:03d}")
    period = meta.get("period", "Quarters 1-40 (10-year horizon)")
    basis = meta.get("basis", "Demonstration scenario (synthetic data)")
    channels = meta.get("channels", [])

    fig, ax = plt.subplots(figsize=(7.2, 9.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(mpatches.Rectangle((0.02, 0.015), 0.96, 0.97, fill=False, linewidth=1.5, edgecolor="black"))

    # -- letterhead -------------------------------------------------------
    ax.text(0.06, 0.955, "RETURN TO SENDER ADVISORY", fontsize=11.5, fontweight="bold")
    ax.text(0.06, 0.936, "Battery-waste externality accounting",
            fontsize=7.5, color="dimgray")
    ax.text(0.06, 0.922, "Student prototype; fictional firm", fontsize=7.5, color="dimgray")
    ax.text(0.94, 0.955, "EXTERNALITY ASSESSMENT", fontsize=11.5, fontweight="bold", ha="right")
    ax.text(0.94, 0.936, f"Ref: {ref}", fontsize=8.5, ha="right", color="dimgray")
    ax.plot([0.06, 0.94], [0.910, 0.910], color="black", linewidth=1.2)

    # -- parties and period ----------------------------------------------
    ax.text(0.06, 0.893, "ILLUSTRATIVE ASSESSED PARTIES", fontsize=8, fontweight="bold", color="dimgray")
    ax.text(0.06, 0.872, "Consumer Lithium-Ion Battery Producers,", fontsize=10)
    ax.text(0.06, 0.852, "via stewardship organization", fontsize=10)
    ax.text(0.56, 0.893, "PREPARED FOR", fontsize=8, fontweight="bold", color="dimgray")
    ax.text(0.56, 0.872, "Provincial Environment & Stewardship", fontsize=10)
    ax.text(0.56, 0.852, "Authority (illustrative client)", fontsize=10)
    ax.text(0.06, 0.822, "PERIOD", fontsize=8, fontweight="bold", color="dimgray")
    ax.text(0.06, 0.802, period, fontsize=10)
    ax.text(0.56, 0.822, "DATA BASIS", fontsize=8, fontweight="bold", color="dimgray")
    ax.text(0.56, 0.802, basis, fontsize=10)

    # -- line-item table --------------------------------------------------
    y = 0.755
    ax.plot([0.06, 0.94], [y + 0.012, y + 0.012], color="black", linewidth=0.8)
    ax.text(0.06, y - 0.008, "LINE ITEM", fontsize=8, fontweight="bold", color="dimgray")
    ax.text(0.72, y - 0.008, "QUANTITY (median)", fontsize=8, fontweight="bold", color="dimgray", ha="right")
    ax.text(0.94, y - 0.008, "AMOUNT (median)", fontsize=8, fontweight="bold", color="dimgray", ha="right")
    y -= 0.040

    tonnes = posterior["total_leaked_tonnage"]
    fires = posterior["expected_fires_total"]
    # landfill = leaked * u_lf particle-by-particle, so median(landfill)/median(leaked) recovers the exact rate
    landfill_rate = posterior["landfill_total"] / tonnes if tonnes > 0 else 0.0
    rows = [
        ("Landfill space & perpetual care",
         f"{tonnes:,.0f} t x {_money(landfill_rate)}/t",
         posterior["landfill_total"]),
        ("Expected fire losses",
         f"{fires:,.1f} expected fires",
         posterior["fire_total"]),
        ("Destroyed material value",
         "spread over sorting cost",
         posterior["material_total"]),
    ]
    for label, qty, amount in rows:
        ax.text(0.06, y, label, fontsize=10)
        ax.text(0.72, y, qty, fontsize=8.5, color="dimgray", ha="right")
        ax.text(0.94, y, _money(amount), fontsize=10, ha="right")
        y -= 0.044

    y += 0.012
    ax.plot([0.06, 0.94], [y, y], color="black", linewidth=0.8)
    y -= 0.036
    ax.text(0.06, y, "ESTIMATED TOTAL (posterior median)", fontsize=11, fontweight="bold")
    ax.text(0.94, y, _money(posterior["total_median"]), fontsize=11, fontweight="bold", ha="right")
    y -= 0.030
    ax.text(0.06, y, "90% credible interval", fontsize=8.5, color="dimgray")
    ax.text(0.94, y, f"{_money(posterior['billable_floor'])} - {_money(posterior['total_p95'])}",
            fontsize=8.5, color="dimgray", ha="right")

    # -- conservative box: the defensible floor -------------------------
    y -= 0.070
    box_h = 0.075
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.06, y - box_h), 0.88, box_h, boxstyle="round,pad=0.008",
        facecolor="#eaf3ea", edgecolor="darkgreen", linewidth=1.4,
    ))
    ax.text(0.09, y - 0.024, "CONSERVATIVE ASSESSMENT FLOOR", fontsize=9, fontweight="bold", color="darkgreen")
    ax.text(0.09, y - 0.055, "5th percentile of modeled cumulative cost; illustrative, not an amount currently due",
            fontsize=7.5, color="darkgreen")
    ax.text(0.91, y - box_h / 2, _money(posterior["billable_floor"]),
            fontsize=15, fontweight="bold", ha="right", va="center", color="darkgreen")

    # -- fee modulation schedule -----------------------------------------
    y -= box_h + 0.055
    ax.text(0.06, y, "FEE MODULATION SCHEDULE (by design class)", fontsize=9, fontweight="bold")
    y -= 0.034
    ax.text(0.09, y, "Embedded (non-removable) battery products", fontsize=9.5)
    ax.text(0.91, y, f"{_money(modulation['embedded_rate'])} / tonne", fontsize=9.5, ha="right")
    y -= 0.030
    ax.text(0.09, y, "Removable battery products", fontsize=9.5)
    ax.text(0.91, y, f"{_money(modulation['removable_rate'])} / tonne", fontsize=9.5, ha="right")
    y -= 0.030
    ax.text(0.09, y, "Modulation ratio (design-for-recovery incentive)", fontsize=9.5)
    ax.text(0.91, y, f"{modulation['ratio']:.2f} : 1", fontsize=9.5, ha="right")

    # -- basis of estimate ------------------------------------------------
    y -= 0.048
    ax.plot([0.06, 0.94], [y + 0.014, y + 0.014], color="lightgray", linewidth=0.8)
    ax.text(0.06, y - 0.004, "BASIS OF ESTIMATE", fontsize=8, fontweight="bold", color="dimgray")
    notes = []
    if channels:
        notes.append("Evidence channels conditioned on: " + ", ".join(channels) + ".")
    notes += [
        "Estimates are Sequential Monte Carlo posteriors over hidden leakage, tonnage, fire intensity and material value.",
        "Each line item is an independently computed posterior median; medians are not additive and need not sum to the total.",
        "Fire cost anchor: continental facility-fire loss data (SOURCES.md [S4]); no per-incident municipal figure exists [U1].",
        "Landfill perpetual-care rate is an ASSUMPTION pending a PSAB landfill-liability figure [U4]; see sensitivity analysis.",
    ]
    yy = y - 0.026
    for note in notes:
        ax.text(0.06, yy, "- " + note, fontsize=7.3, color="dimgray")
        yy -= 0.020

        ax.text(0.5, 0.030,
            "SYNTHETIC DEMONSTRATION · NOT AN INVOICE · NOT FINANCIAL OR LEGAL ADVICE",
            fontsize=7.5, ha="center", color="firebrick", fontweight="bold")

    fig.tight_layout()
    return fig
