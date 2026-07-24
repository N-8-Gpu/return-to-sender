"""app.py -- Streamlit demo, framed as a consulting engagement.

Return to Sender Advisory (a fictional student-prototype firm) helps a
government client measure the hidden leakage of lithium-ion batteries into the
waste stream, price the resulting externalities, and bill them to producers
through an evidence-backed, legally defensible invoice.

Single page, five tabs: Executive brief / Evidence & estimator / The invoice /
Legal pathway / Methodology & sources. Two data modes: a synthetic demo
scenario, or an uploaded quarterly-observations CSV (a real engagement's data).
"""

from dataclasses import dataclass, replace
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit import runtime

import invoice as inv
from config import Config
from filter import run_filter
from simulator import (
    observations_csv_template,
    parse_observations_csv,
    simulate,
)

N_PARTICLES = 5000
INK = "#202622"
MUTED = "#66716a"
PAPER = "#f7f7f2"
GREEN = "#176b4d"
GREEN_LIGHT = "#dcebe2"
AMBER = "#c66a15"
AMBER_LIGHT = "#f2dcc5"

CHANNEL_INFO = {
    "audits": ("Waste-composition audits",
               "Sampled garbage-bag audits per design class. Real-world analog: Tetra Tech "
               "waste-composition studies already commissioned by provinces [S12]."),
    "depots": ("Depot intake reports",
               "Quarterly tonnage properly returned through collection depots. Analog: "
               "Call2Recycle program reporting [S11]."),
    "fires": ("Facility fire incidents",
              "Quarterly battery-fire counts at waste facilities. Analog: Calgary landfill "
              "fire logs (~40/yr) [S1][S2]."),
    "prices": ("Commodity price monitor",
               "Recyclate-vs-virgin material price spread. Analog: published commodity indices."),
    "mailbox": ("Mail-back pilot",
                "Prepaid mail-back returns; activates partway through the horizon and ramps up. "
                "Analog: Call2Recycle's Recycle Your Vapes mail program [S11]."),
    "compactor": ("Compactor sensors",
                  "High-frequency detection counts at compaction; the information-rich channel. "
                  "Analog: acoustic/thermal battery detection (cf. Fire Rover [S15])."),
    "tags": ("Unit-level tags",
             "Stylized battery-passport data: a direct but noisy read on leakage. The EU mandates "
             "passports for larger batteries from 2027; small consumer cells are the gap [S10]."),
}

GUIDED_SEED = 0


@dataclass(frozen=True)
class GuidedStage:
    """One deterministic beat in the presenter-facing demo."""

    title: str
    short_title: str
    description: str
    active_channels: tuple[str, ...]
    t_deposit: int | None = None


GUIDED_STAGES = {
    "sparse": GuidedStage(
        title="1. Sparse evidence",
        short_title="Sparse evidence",
        description="A quarterly waste audit gives us a signal, but leaves a wide defensible range.",
        active_channels=("audits",),
    ),
    "network": GuidedStage(
        title="2. Evidence network",
        short_title="Evidence network",
        description="Existing records and practical sensors combine into a much sharper estimate.",
        active_channels=("audits", "depots", "fires", "prices", "mailbox", "compactor"),
    ),
    "deposit": GuidedStage(
        title="3. Deposit-return scenario",
        short_title="Deposit-return scenario",
        description="The same evidence network estimates the response after a modeled program begins.",
        active_channels=("audits", "depots", "fires", "prices", "mailbox", "compactor"),
        t_deposit=20,
    ),
}


def guided_channels(stage_key: str) -> dict[str, bool]:
    """Return a complete filter-channel mapping for a guided stage."""
    enabled = set(GUIDED_STAGES[stage_key].active_channels)
    return {key: key in enabled for key in CHANNEL_INFO}


def mean_credible_band_width(engagement: dict) -> float:
    """Mean width of the quarterly 90% leakage credible interval."""
    quantiles = engagement["results"]["quantiles"]["phi_bar"]
    return float(np.mean(quantiles[:, 2] - quantiles[:, 0]))


def relative_uncertainty_reduction(baseline_width: float, current_width: float) -> float:
    """Fractional band-width reduction; negative values honestly report widening."""
    if baseline_width <= 0:
        return 0.0
    return (baseline_width - current_width) / baseline_width


def modeled_avoided_externality(engagement: dict) -> float | None:
    """Median no-program cost less median intervention cost, when paired data exist."""
    baseline = engagement.get("baseline_posterior")
    if baseline is None:
        return None
    return float(baseline["total_median"] - engagement["posterior"]["total_median"])


def run_guided_stage(stage_key: str) -> dict:
    """Run one fixed guided stage through the same cached demo pipeline."""
    stage = GUIDED_STAGES[stage_key]
    channels = tuple(sorted(guided_channels(stage_key).items()))
    return run_demo_engagement(GUIDED_SEED, stage.t_deposit, 0.0, channels)


def apply_demo_styles() -> None:
    """Apply a restrained, projector-friendly layer over Streamlit's base theme."""
    st.markdown(
        """
        <style>
        .stApp { background: #f7f7f2; color: #202622; }
        [data-testid="stHeader"] { background: rgba(247, 247, 242, 0.92); }
        [data-testid="stSidebar"] { background: #202622; }
        [data-testid="stSidebar"] * { color: #f7f7f2; }
        [data-testid="stSidebar"] .stButton button {
            background: #f7f7f2; color: #202622; border-color: #f7f7f2;
        }
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
        h1, h2, h3 { font-family: "Avenir Next", Avenir, "Helvetica Neue", sans-serif; letter-spacing: 0; }
        h1 { font-size: clamp(2rem, 4vw, 3.35rem); line-height: 1.02; max-width: 900px; }
        h2 { font-size: 1.45rem; }
        p, li, label, [data-testid="stMetricLabel"] { font-family: Georgia, "Times New Roman", serif; }
        [data-testid="stMetric"] {
            background: #ffffff; border-top: 3px solid #176b4d; padding: 1rem 1.1rem;
        }
        [data-testid="stMetricValue"] { font-family: "Avenir Next", Avenir, sans-serif; color: #202622; }
        .stage-kicker { color: #176b4d; font: 700 0.8rem "Avenir Next", sans-serif; text-transform: uppercase; }
        .demo-disclosure {
            border-left: 4px solid #c66a15; background: #fff; padding: 0.75rem 1rem; margin: 1rem 0 1.5rem;
            font: 0.92rem Georgia, serif;
        }
        .stage-copy { font-size: 1.15rem; max-width: 760px; color: #465049; }
        .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid #cdd2cd; }
        .stTabs [data-baseweb="tab"] { font-family: "Avenir Next", Avenir, sans-serif; font-weight: 650; }
        .stButton button { border-radius: 4px; min-height: 2.8rem; font-weight: 650; }
        .stDownloadButton button { border-radius: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fmt_money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x/1e6:,.1f}M"
    return f"${x:,.0f}"


def _make_config(seed: int, t_deposit: int | None, tau: float, n_quarters: int | None = None) -> Config:
    cfg = Config(seed=seed)
    cfg.phi_dynamics.t_deposit = t_deposit
    cfg.tag.tau = tau
    if n_quarters is not None:
        cfg.geography.n_quarters = n_quarters
    return cfg


@st.cache_data(show_spinner=False)
def run_demo_engagement(seed: int, t_deposit: int | None, tau: float,
                        channels: tuple[tuple[str, bool], ...]) -> dict:
    """Simulate the synthetic scenario and run the filter; if the deposit is on,
    also run the no-deposit counterfactual with the same latent noise (the truth
    process consumes an identical draw sequence, so the comparison is paired)."""
    active = dict(channels)
    cfg = _make_config(seed, t_deposit, tau)
    scenario = simulate(cfg, np.random.default_rng(seed))
    results = run_filter(scenario["obs"], active, cfg, n_particles=N_PARTICLES, seed=seed)
    posterior = inv.compute_cost_posterior(results, cfg)

    baseline_posterior = None
    if t_deposit is not None:
        cfg_base = _make_config(seed, None, tau)
        scenario_base = simulate(cfg_base, np.random.default_rng(seed))
        results_base = run_filter(scenario_base["obs"], active, cfg_base, n_particles=N_PARTICLES, seed=seed)
        baseline_posterior = inv.compute_cost_posterior(results_base, cfg_base)

    return {
        "mode": "demo",
        "cfg": cfg,
        "truth": scenario["truth"],
        "obs": scenario["obs"],
        "results": results,
        "posterior": posterior,
        "baseline_posterior": baseline_posterior,
        "active": active,
    }


@st.cache_data(show_spinner=False)
def run_upload_engagement(csv_text: str, seed: int, t_deposit: int | None, tau: float,
                          channels: tuple[tuple[str, bool], ...]) -> dict:
    """Parse client data and run the filter on it. No ground truth exists here,
    so truth overlays and the counterfactual are unavailable."""
    active = dict(channels)
    obs, T, sales = parse_observations_csv(csv_text)
    t_dep = None if t_deposit is None else min(t_deposit, T - 1)
    cfg = _make_config(seed, t_dep, tau, n_quarters=T)
    if sales is not None:
        cfg.geography.sales_override = tuple(float(x) for x in sales)
    results = run_filter(obs, active, cfg, n_particles=N_PARTICLES, seed=seed)
    posterior = inv.compute_cost_posterior(results, cfg)
    return {
        "mode": "upload",
        "cfg": cfg,
        "truth": None,
        "obs": obs,
        "results": results,
        "posterior": posterior,
        "baseline_posterior": None,
        "active": active,
    }


def bands_chart(eng: dict) -> plt.Figure:
    cfg, results, truth = eng["cfg"], eng["results"], eng["truth"]
    T = cfg.geography.n_quarters
    t_axis = np.arange(T)
    q = results["quantiles"]["phi_bar"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.fill_between(t_axis, q[:, 0], q[:, 2], color=GREEN_LIGHT, alpha=0.8,
                    label="90% credible band")
    ax.plot(t_axis, q[:, 1], color=GREEN, linewidth=2.2, label="Estimated leakage")
    if truth is not None:
        ax.plot(t_axis, truth["phi_bar"], linestyle="--", color="black", label="True leakage (synthetic)")
        ax.axvline(cfg.regime.t_shock, color="gray", linestyle=":", linewidth=1.2)
        ax.annotate("market shock", (cfg.regime.t_shock, ax.get_ylim()[1]), fontsize=9,
                    color="gray", rotation=90, va="top", ha="right")
    if cfg.phi_dynamics.t_deposit is not None:
        ax.axvline(cfg.phi_dynamics.t_deposit, color="green", linestyle=":", linewidth=1.2)
        ax.annotate("program begins", (cfg.phi_dynamics.t_deposit, ax.get_ylim()[1]), fontsize=9,
                    color="green", rotation=90, va="top", ha="right")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Leakage fraction")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    fig.tight_layout()
    return fig


def evidence_comparison_chart(sparse: dict, network: dict) -> plt.Figure:
    """Compare uncertainty under sparse evidence and the guided evidence network."""
    sparse_q = sparse["results"]["quantiles"]["phi_bar"]
    network_q = network["results"]["quantiles"]["phi_bar"]
    truth = sparse["truth"]["phi_bar"]
    t_axis = np.arange(len(truth))
    reduction = relative_uncertainty_reduction(
        mean_credible_band_width(sparse), mean_credible_band_width(network)
    )

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(t_axis, sparse_q[:, 0], sparse_q[:, 2], color=AMBER_LIGHT, alpha=0.7,
                    label="Audits only: 90% range")
    ax.fill_between(t_axis, network_q[:, 0], network_q[:, 2], color=GREEN_LIGHT, alpha=0.95,
                    label="Evidence network: 90% range")
    ax.plot(t_axis, network_q[:, 1], color=GREEN, linewidth=2.2,
            label="Evidence-network estimate")
    ax.plot(t_axis, truth, color=INK, linestyle="--", linewidth=1.2,
            label="Synthetic truth (hidden from estimator)")
    ax.text(
        0.02, 0.05, f"Average uncertainty range reduced {reduction:.0%}",
        transform=ax.transAxes, fontsize=11, fontweight="bold", color=GREEN,
        bbox={"boxstyle": "square,pad=0.45", "facecolor": "white", "edgecolor": GREEN},
    )
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Share leaking into garbage")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9.5, frameon=False)
    fig.tight_layout()
    return fig


def cumulative_cost_chart(eng: dict) -> plt.Figure:
    post = eng["posterior"]
    T = len(post["cumulative_median"])
    t_axis = np.arange(T)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(t_axis, post["cumulative_p05"], post["cumulative_p95"],
                    color=GREEN_LIGHT, alpha=0.8, label="90% credible band")
    ax.plot(t_axis, post["cumulative_median"], color=GREEN, linewidth=2.3,
            label="Current scenario (median)")
    ax.plot(t_axis, post["cumulative_p05"], color=GREEN, linewidth=1.2, linestyle="--",
            label="Conservative floor (5th pct)")
    baseline = eng.get("baseline_posterior")
    if baseline is not None:
        ax.plot(t_axis, baseline["cumulative_median"], color=AMBER, linewidth=2,
                label="No-program scenario (median)")
        ax.fill_between(t_axis, post["cumulative_median"], baseline["cumulative_median"],
                        color=AMBER_LIGHT, alpha=0.45, label="Modeled avoided externality")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Cumulative cost ($)")
    ax.legend(loc="upper left", fontsize=9.5, frameon=False)
    fig.tight_layout()
    return fig


def n_eff_chart(eng: dict) -> plt.Figure:
    n_eff = eng["results"]["n_eff"] / N_PARTICLES
    fig, ax = plt.subplots(figsize=(9, 1.8))
    ax.plot(np.arange(len(n_eff)), n_eff, color=GREEN)
    ax.axhline(0.5, color=AMBER, linestyle=":", linewidth=1.3,
               label="Refresh particles below this line")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Particle fit")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


def render_guided_invoice(eng: dict) -> None:
    """Render the assessment, downloads, and optional assumption sweep."""
    cfg, post = eng["cfg"], eng["posterior"]
    T = cfg.geography.n_quarters
    modulation = inv.modulation_table(post["total_median"], post["total_leaked_tonnage"], cfg)
    meta = {
        "ref": f"RTS-DEMO-{cfg.seed:03d}",
        "period": f"Quarters 1-{T} ({T/4:.0f}-year horizon)",
        "basis": "Synthetic scenario calibrated to cited anchors",
        "channels": [key for key, active in eng["active"].items() if active],
    }
    invoice_fig = inv.render_invoice(post, modulation, seed=cfg.seed, meta=meta)
    st.pyplot(invoice_fig)

    png_buffer = BytesIO()
    pdf_buffer = BytesIO()
    invoice_fig.savefig(png_buffer, format="png", dpi=200, bbox_inches="tight")
    invoice_fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
    download_png, download_pdf, _ = st.columns([1, 1, 2])
    download_png.download_button(
        "Download PNG", png_buffer.getvalue(),
        file_name=f"externality-assessment-{meta['ref']}.png", mime="image/png",
    )
    download_pdf.download_button(
        "Download PDF", pdf_buffer.getvalue(),
        file_name=f"externality-assessment-{meta['ref']}.pdf", mime="application/pdf",
    )

    with st.expander("Test the cost assumptions"):
        st.caption(
            "The municipal cost per fire and landfill perpetual-care rate are not yet locally "
            "verified. Change either assumption; the same leakage posterior is reused."
        )
        u_f = st.slider("Cost per fire incident ($)", 50_000, 2_000_000, int(cfg.invoice.u_f),
                        step=50_000, key="guided_sens_uf")
        u_lf = st.slider("Landfill perpetual care ($/tonne)", 50, 500, int(cfg.invoice.u_lf),
                         step=10, key="guided_sens_ulf")
        cfg_sens = replace(cfg, invoice=replace(cfg.invoice, u_f=float(u_f), u_lf=float(u_lf)))
        post_sens = inv.compute_cost_posterior(eng["results"], cfg_sens)
        sens_median, sens_floor, sens_fire = st.columns(3)
        sens_median.metric("Median estimate", _fmt_money(post_sens["total_median"]))
        sens_floor.metric("Conservative floor", _fmt_money(post_sens["billable_floor"]))
        sens_fire.metric("Fire-loss estimate", _fmt_money(post_sens["fire_total"]))


def render_guided_app() -> None:
    """Render the deterministic, presenter-facing three-stage experience."""
    st.markdown('<div class="stage-kicker">Return to Sender Advisory / Live prototype</div>',
                unsafe_allow_html=True)
    st.title("Make hidden battery waste costs visible")
    st.caption("A synthetic proof of concept for provincial stewardship decisions")
    st.markdown(
        '<div class="demo-disclosure"><strong>Demonstration basis.</strong> The scenario is '
        'synthetic, seeded for reproducibility, and calibrated to cited public anchors. It is '
        'not a real invoice or a claim of legal authority.</div>',
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("guided_stage", "sparse")
    st.markdown("**Move through the three demo stages**")
    stage_columns = st.columns(3)
    for column, (stage_key, stage) in zip(stage_columns, GUIDED_STAGES.items()):
        if column.button(
            stage.title,
            key=f"guided_stage_{stage_key}",
            type="primary" if st.session_state["guided_stage"] == stage_key else "secondary",
            use_container_width=True,
        ):
            st.session_state["guided_stage"] = stage_key

    stage_key = st.session_state["guided_stage"]
    stage = GUIDED_STAGES[stage_key]
    with st.spinner(f"Loading {stage.short_title.lower()}..."):
        eng = run_guided_stage(stage_key)

    st.markdown(f'<div class="stage-kicker">Current stage / {stage.title}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<p class="stage-copy">{stage.description}</p>', unsafe_allow_html=True)

    post = eng["posterior"]
    avoided = modeled_avoided_externality(eng)
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Estimated leaked mass", f"{post['total_leaked_tonnage']:,.0f} t")
    metric_2.metric("Median externality", _fmt_money(post["total_median"]))
    metric_3.metric("Conservative floor", _fmt_money(post["billable_floor"]))
    if avoided is None:
        metric_4.metric("Expected fire incidents", f"{post['expected_fires_total']:,.0f}")
    else:
        metric_4.metric("Modeled avoided cost", _fmt_money(avoided))

    tab_case, tab_evidence, tab_invoice, tab_appendix = st.tabs(
        ["1. The case", "2. Evidence", "3. Externality assessment", "Appendix"]
    )

    with tab_case:
        if stage_key == "sparse":
            st.markdown(
                "### The damage is real; its scale is hard to observe\n"
                "Batteries discarded in ordinary waste can start facility fires, consume landfill "
                "capacity, and destroy recoverable material value. A single audit program detects "
                "the problem, but leaves a wide range around how much is leaking."
            )
        elif stage_key == "network":
            st.markdown(
                "### Existing records become an evidence network\n"
                "Waste audits, depot intake, fire logs, market prices, mail-back returns, and "
                "compactor detections describe the same hidden flow from different angles. The "
                "model combines them without pretending any one source is perfect."
            )
        else:
            st.markdown(
                "### A program can be tested against the no-program scenario\n"
                "At quarter 20, the model introduces a deposit-return program. The comparison uses "
                "the same seeded latent noise in both scenarios, isolating the modeled program "
                "response without presenting it as empirical causal proof."
            )
        st.markdown("**Cumulative externality over the ten-year scenario**")
        st.pyplot(cumulative_cost_chart(eng))

    with tab_evidence:
        if stage_key == "sparse":
            st.markdown(
                "### One signal leaves room for dispute\n"
                "The shaded region is the model's 90% credible range. The dashed synthetic truth is "
                "shown to us for validation, but remains hidden from the estimator."
            )
            st.pyplot(bands_chart(eng))
            st.metric("Average 90% uncertainty range", f"{mean_credible_band_width(eng):.1%}")
        else:
            sparse_eng = run_guided_stage("sparse")
            network_eng = run_guided_stage("network")
            sparse_width = mean_credible_band_width(sparse_eng)
            network_width = mean_credible_band_width(network_eng)
            reduction = relative_uncertainty_reduction(sparse_width, network_width)
            st.markdown(
                "### More independent evidence sharpens the defensible range\n"
                "Both estimates face the same synthetic reality. The only change is which evidence "
                "channels the estimator is allowed to use."
            )
            st.pyplot(evidence_comparison_chart(sparse_eng, network_eng))
            evidence_1, evidence_2, evidence_3 = st.columns(3)
            evidence_1.metric("Audits-only range", f"{sparse_width:.1%}")
            evidence_2.metric("Network range", f"{network_width:.1%}")
            evidence_3.metric("Range reduction", f"{reduction:.0%}")

        active_list = [CHANNEL_INFO[key][0] for key, active in eng["active"].items() if active]
        st.markdown("**Evidence in this stage:** " + ", ".join(active_list))
        with st.expander("How the estimator works, in four verbs"):
            st.markdown(
                "**Guess:** advance 5,000 plausible hidden states.  "
                "**Weight:** score each against the observed evidence.  "
                "**Normalize:** turn scores into probabilities.  "
                "**Refresh:** replace implausible particles when the cloud becomes too concentrated."
            )

    with tab_invoice:
        st.markdown(
            "### Turn an estimate into a transparent assessment\n"
            "The conservative floor is the 5th percentile of cumulative modeled cost. It is shown "
            "alongside the full range and every material assumption, not as a real amount due."
        )
        render_guided_invoice(eng)

    with tab_appendix:
        st.markdown(
            "### What is proven, modeled, and still unknown\n"
            "**Proven in tests:** the filter tracks stable truth, detects a market regime change, "
            "and its 90% interval has empirical coverage across seeded scenarios.\n\n"
            "**Modeled:** a single city, two battery design classes, 40 quarters, and a deposit "
            "response calibrated to a cited capture-rate ceiling.\n\n"
            "**Still open:** Calgary-specific fire cost, landfill perpetual-care cost, and authority "
            "for damage-based producer fees. These remain labeled assumptions or legal questions."
        )
        st.markdown("**Filter health**")
        st.pyplot(n_eff_chart(eng))
        with st.expander("Source register"):
            st.markdown(
                "- **S1-S3:** Calgary fire frequency and battery-crushing mechanism.\n"
                "- **S4-S7:** continental facility-fire costs and disruption.\n"
                "- **S10-S12:** battery passports, stewardship reporting, and waste audits.\n"
                "- **S13-S14:** Nova Scotia deposit-return performance and EPR pathway.\n"
                "- **U1, U3, U4:** local fire cost, legal authority, and landfill cost remain open.\n\n"
                "Full citations and links are maintained in `SOURCES.md`."
            )
        with st.expander("Producer-level reconciliation roadmap"):
            st.markdown(
                "The simulator includes brand tallies among proper returns. A future version could "
                "estimate producer-specific leakage after real sales and return data are available; "
                "v1 does not use these tallies in the filter."
            )


def render_advanced_app() -> None:
    """Render the page. Split out so importing app.py (e.g. from tests) is side-effect-free."""
    # ============================= sidebar =====================================

    with st.sidebar:
        mode = st.radio("Data source", ["Demo mode (synthetic scenario)", "Client data upload (CSV)"],
                        key="mode_radio")
        upload_mode = mode.startswith("Client")

        csv_text = None
        if upload_mode:
            template_cfg = Config()
            st.download_button(
                "Download CSV template",
                observations_csv_template(template_cfg),
                file_name="observations_template.csv",
                mime="text/csv",
                help="Quarterly rows; recognized channel columns; blank cells = no reading. "
                     "The sales column is the disclosed sales series (required in every row "
                     "if present); without it, dollar figures use the synthetic sales config.",
            )
            uploaded = st.file_uploader("Quarterly observations CSV", type=["csv"], key="uploader")
            if uploaded is not None:
                try:
                    csv_text = uploaded.getvalue().decode("utf-8")
                except UnicodeDecodeError:
                    st.error("The file isn't UTF-8 text. Export it as a plain CSV and retry.")

        st.header("Evidence programs")
        active_channels = {}
        for key, (label, help_text) in CHANNEL_INFO.items():
            default = key == "audits"
            active_channels[key] = st.checkbox(label, value=default, key=f"ch_{key}", help=help_text)
        tau = st.slider("Tag adoption (tau)", 0.0, 0.3, 0.15, step=0.01,
                        disabled=not active_channels["tags"],
                        help="Share of units carrying tags; higher adoption = less noisy readings.")

        st.header("Deposit intervention")
        deposit_on = st.checkbox("Deposit-return program", value=False, key="deposit_on",
                                 help="Modeled on Nova Scotia's ~80% deposit-return capture [S13].")
        t_deposit_input = st.slider("Program start quarter", 0, 39, 20, disabled=not deposit_on,
                                    key="t_dep")

        st.header("Estimator")
        seed = int(st.number_input("Seed", min_value=0, value=0, step=1, key="seed"))
        run_clicked = st.button("Run analysis", type="primary", use_container_width=True)

    # ============================= engagement run ==============================

    t_deposit = t_deposit_input if deposit_on else None
    tau_eff = tau if active_channels["tags"] else 0.0
    channels_key = tuple(sorted(active_channels.items()))
    settings_key = (mode, seed, t_deposit, tau_eff, channels_key, csv_text)

    if run_clicked:
        if upload_mode and csv_text is None:
            st.sidebar.error("Upload a CSV first (or switch to demo mode).")
        else:
            try:
                with st.spinner("Running estimator..."):
                    if upload_mode:
                        eng = run_upload_engagement(csv_text, seed, t_deposit, tau_eff, channels_key)
                    else:
                        eng = run_demo_engagement(seed, t_deposit, tau_eff, channels_key)
                st.session_state["engagement"] = eng
                st.session_state["engagement_settings"] = settings_key
            except ValueError as exc:
                st.sidebar.error(str(exc))

    # ============================= main page ===================================

    st.title("Hidden battery leakage: measured, priced, billed")
    st.caption(
        "Return to Sender Advisory | Prepared for the Provincial Environment & Stewardship "
        "Authority (illustrative client) | Synthetic demonstration; not financial or legal advice"
    )

    if "engagement" not in st.session_state:
        st.markdown(
            """
    **The engagement.** Lithium-ion batteries leak into garbage streams, where they start fires,
    consume landfill capacity, and destroy recoverable material value. Operators and municipalities
    carry those costs today; producers do not [S6]. This tool estimates the *hidden* leakage rate
    from whatever evidence programs the client runs, prices the externality with quantified
    uncertainty, and produces an invoice whose **billable floor** (the 5th percentile of cumulative
    cost) is designed to survive challenge.

    **To begin:** choose evidence programs in the sidebar and click **Run analysis**. Demo mode uses
    a fully synthetic ten-year scenario; upload mode accepts your own quarterly observations CSV.

    **The demonstration argument.** Turn programs on one at a time and watch the uncertainty band
    around the leakage estimate shrink. Every program the client commissions doesn't just collect
    batteries; it can tighten the range available for a proportionate, evidence-based assessment.
            """
        )
        st.stop()

    eng = st.session_state["engagement"]
    if st.session_state.get("engagement_settings") != settings_key:
        st.warning("Sidebar settings have changed since this analysis ran. Click **Run analysis** to refresh.")

    cfg, post = eng["cfg"], eng["posterior"]
    T = cfg.geography.n_quarters
    active_list = [CHANNEL_INFO[k][0] for k, v in eng["active"].items() if v]
    is_demo = eng["mode"] == "demo"
    basis_label = "Demonstration scenario (synthetic data)" if is_demo else "Client data upload"

    tab_exec, tab_evidence, tab_invoice, tab_legal, tab_method = st.tabs(
        ["Executive brief", "Evidence & estimator", "The invoice", "Legal pathway", "Methodology & sources"]
    )

    # --------------------------- Executive brief -------------------------------
    with tab_exec:
        band_rel = (post["total_p95"] - post["billable_floor"]) / post["total_median"] if post["total_median"] else 0.0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Leaked tonnage (median)", f"{post['total_leaked_tonnage']:,.0f} t")
        c2.metric("Externality estimate (median)", _fmt_money(post["total_median"]))
        c3.metric("Conservative floor (illustrative)", _fmt_money(post["billable_floor"]))
        c4.metric("Expected fire incidents", f"{post['expected_fires_total']:,.0f}")

        if eng["baseline_posterior"] is not None:
            avoided = eng["baseline_posterior"]["total_median"] - post["total_median"]
            st.metric("Avoided externality attributable to the deposit program (paired counterfactual)",
                      _fmt_money(max(avoided, 0.0)),
                      help="Median externality without the program minus median with it, on the "
                           "same latent noise sequence.")

        st.markdown(
            f"""
    **Finding.** Over {T} quarters, an estimated **{post['total_leaked_tonnage']:,.0f} tonnes** of
    lithium-ion batteries left the proper recovery stream. The resulting externality (landfill
    liability, expected fire losses, destroyed material value) has a posterior median of
    **{_fmt_money(post['total_median'])}**, of which **{_fmt_money(post['billable_floor'])}** is
    defensible under conservative assumptions (5th percentile).

    **Recommendation.** Use the conservative floor as a planning benchmark while local costs and
    legal authority are validated, with design-class modulation shown as a policy option. Each
    additional evidence program can narrow the credible interval (currently spanning
    {band_rel:.0%} of the median estimate) and change the floor when the observations support it;
    the value of measurement is a more proportionate and transparent assessment.
            """
        )
        st.markdown("**Exhibit A: cumulative externality and the billable floor**")
        st.pyplot(cumulative_cost_chart(eng))

    # --------------------------- Evidence & estimator --------------------------
    with tab_evidence:
        st.markdown(
            "**Exhibit B: hidden leakage rate, posterior vs. evidence.** The shaded band is the "
            "90% credible interval; each active evidence program tightens it."
            + (" The dashed line is the synthetic ground truth the estimator cannot see."
               if is_demo else "")
        )
        st.pyplot(bands_chart(eng))
        st.markdown(f"**Evidence conditioned on:** {', '.join(active_list) if active_list else 'none'}")

        with st.expander("Live demonstration script (4 beats)"):
            st.markdown(
                """
    1. **Audits only.** Point at the wide band: with one evidence program, the province is
       nearly guessing, and so is any fee it tries to defend.
    2. **Add the mail-back pilot, then compactor sensors.** The band visibly narrows around
       the same underlying truth. More evidence, same reality, sharper number.
     3. **Switch on the deposit program.** Leakage bends downward a few quarters later;
         after being given the program start, the estimator uses evidence to estimate the response.
     4. **Open the invoice tab.** The evidence programs did not change the synthetic reality;
         they tightened the range used to calculate a conservative illustrative floor.
                """
            )

        if is_demo:
            with st.expander("Exhibit C: brand mass-balance reconciliation (roadmap)"):
                st.markdown(
                    "Plant-level brand tallies among proper returns, from the simulator only "
                    "(not conditioned on by the estimator in v1). Roadmap: per-brand leakage "
                    "posteriors would let the invoice be split by producer, not just design class."
                )
                tally = eng["obs"]["brand_tally"]
                fig, ax = plt.subplots(figsize=(9, 2.8))
                for i, name in enumerate(cfg.brand.brand_names):
                    ax.plot(np.arange(T), tally[:, i], label=name)
                ax.set_xlabel("Quarter")
                ax.set_ylabel("Units among proper returns")
                ax.legend(fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)

    # --------------------------- The invoice -----------------------------------
    with tab_invoice:
        modulation = inv.modulation_table(post["total_median"], post["total_leaked_tonnage"], cfg)
        meta = {
            "ref": f"RTS-2026-{cfg.seed:03d}",
            "period": f"Quarters 1-{T} ({T/4:.0f}-year horizon)",
            "basis": basis_label,
            "channels": [k for k, v in eng["active"].items() if v],
        }
        invoice_fig = inv.render_invoice(post, modulation, seed=cfg.seed, meta=meta)
        st.pyplot(invoice_fig)

        buf = BytesIO()
        invoice_fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        st.download_button("Download invoice (PNG)", buf.getvalue(),
                           file_name=f"externality-invoice-{meta['ref']}.png", mime="image/png")

        with st.expander("Sensitivity: unit-cost assumptions (SOURCES.md rule: hypotheses get sweeps)"):
            st.caption(
                "Cost per fire is anchored to continental facility-fire data [S4]; no municipal "
                "per-incident figure exists [U1]. Landfill perpetual care is an ASSUMPTION [U4]. "
                "Sweep both; the invoice recomputes from the same posterior."
            )
            u_f = st.slider("Cost per fire incident ($)", 50_000, 2_000_000, int(cfg.invoice.u_f),
                            step=50_000, key="sens_uf")
            u_lf = st.slider("Landfill perpetual care ($/tonne)", 50, 500, int(cfg.invoice.u_lf),
                             step=10, key="sens_ulf")
            cfg_sens = replace(cfg, invoice=replace(cfg.invoice, u_f=float(u_f), u_lf=float(u_lf)))
            post_sens = inv.compute_cost_posterior(eng["results"], cfg_sens)
            s1, s2, s3 = st.columns(3)
            s1.metric("Median estimate", _fmt_money(post_sens["total_median"]))
            s2.metric("Billable floor", _fmt_money(post_sens["billable_floor"]))
            s3.metric("Fire line item", _fmt_money(post_sens["fire_total"]))

    # --------------------------- Legal pathway ---------------------------------
    with tab_legal:
        st.markdown(
            """
    ### Who can levy this invoice, and under what authority

    **Not legal advice.** This is a student prototype's map of the regulatory terrain, built from
    the cited sources; every open question is labeled as one.

    **1. The machinery already exists.** Extended producer responsibility (EPR) for batteries is
    provincial jurisdiction in Canada, administered through producer responsibility organizations
    (PROs). Call2Recycle operates as the battery PRO in Ontario and Alberta and runs programs in
    six provinces [S11]. Nova Scotia's EPR framework took effect December 1, 2025, explicitly
    shifting program costs from municipalities to producers, and its 2026 producer-fee
    consultation is underway [S14]. **The invoice's addressee ("producers, via stewardship
    organization") is an existing legal person with an existing fee mechanism.**

    **2. What's missing is the price, not the pipe.** Today's EPR fees recover *program costs*
    (collection, transport, processing). They do not price fires, landfill perpetual-care
    liability, or destroyed material value: the externalities in this invoice. Nobody measures
    them, so nobody bills them; the absence of a municipal per-fire cost figure [U1] is itself
    the finding. Meanwhile the burden lands on operators and municipalities: ~$2.5B in North
    American facility fire losses in 2025 [S4], one in four facility battery fires causing
    service disruption and six-figure damage [S5], and industry's own assessment that
    "producers get off scot-free" [S6].

    **3. Design-modulated fees have precedent; damage-based fees are the open question.**
    Fee modulation by product design (the embedded-vs-removable split on this invoice) is an
    established EPR concept: charging harder-to-recover designs more is an incentive, not a
    penalty. Whether a province's current statutes authorize fees calibrated to *measured
    downstream damage*, rather than program cost recovery, is **unverified and flagged as an
    open legal question [U3]**. Two candidate pathways, in order of ambition:
    - *Fee-schedule modulation within existing PRO authority:* reweight existing fees using the
      measured externality ratio between design classes.
    - *Regulatory amendment adding externality cost recovery:* following the template Nova
      Scotia set by shifting whole cost categories onto producers [S14].

    **4. Why the estimator is the legal strategy.** A fee challenged in consultation or court
    must be evidence-based and proportionate. That is precisely what the posterior provides:
    the **billable floor is the 5th percentile**: the amount defensible *even if* the estimate
    is substantially wrong in producers' favor. Uncertainty is disclosed, not hidden; each new
    evidence program narrows the interval and raises the floor. The audit trail (which channels,
    which likelihoods, which assumptions; see Methodology) is the exhibit list.

    **5. Tailwinds.** Federal Landfill Methane Regulations already put landfill externalities on
    a regulatory footing [S8][S9]. The EU's battery-passport mandate (Feb 2027) makes unit-level
    producer traceability law for larger batteries, while explicitly excluding small consumer
    cells [S10]: exactly the gap this program addresses, and the reason the tags channel exists
    in the model.
            """
        )

    # --------------------------- Methodology & sources -------------------------
    with tab_method:
        if is_demo:
            data_note = "demo scenario"
        elif cfg.geography.sales_override is not None:
            data_note = (
                "client upload with a disclosed sales column; pre-sample sales history is held "
                "flat at the earliest observed level (assumption)"
            )
        else:
            data_note = (
                "client upload without a sales column, so the tonnage anchor still uses the "
                "configured synthetic sales series. Add a `sales` column for real dollar figures"
            )
        st.markdown(
            f"""
    ### Estimator

    Sequential Monte Carlo (particle) filter, {N_PARTICLES:,} particles, {T} quarterly steps.
    Hidden states: end-of-life tonnage (log-space, anchored to a disclosed sales series through a
    lifespan kernel), per-design-class leakage fractions (logit-scale random walks, shifted by the
    deposit intervention), a two-state market regime, the recyclate-virgin price spread
    (regime-switching AR(1)), and log fire intensity per leaked tonne. Each active evidence
    channel contributes an exact likelihood (binomial audits, lognormal depot tonnage, Poisson
    fires/mail-back/compactor counts, Gaussian prices and tags). Systematic resampling when
    N_eff < N/2. Full equations and every parameter: `config.py` and `simulator.py`.

    **Cost model.** Per quarter: `C_t = leakage x tonnage x (landfill rate + fire intensity x cost
    per fire + max(price spread - sorting cost, 0))`, accumulated over the horizon; the invoice
    reports posterior medians, the 90% credible interval, and the 5th-percentile billable floor.

    **Honesty ledger.** Synthetic sales series and geography ({data_note}); the tags channel is a
    stylized stand-in for battery-passport data [S10]; fire
    cost per incident uses a continental anchor, not municipal data [S4][U1]; landfill
    perpetual-care rate is an assumption pending a PSAB landfill-liability figure [U4]. Every
    unsourced constant is tagged ASSUMPTION in `config.py` and swept in the sensitivity panel.
            """
        )
        st.markdown("**Exhibit D: filter health (effective sample size).** "
                    "N_eff near N means the particle cloud matches the evidence; collapses would "
                    "flag model misfit. Dips trigger systematic resampling.")
        st.pyplot(n_eff_chart(eng))
        st.markdown(
            """
    ### Source register (see SOURCES.md for full citations)

    | Tag | Anchors | Feeds |
    |---|---|---|
    | S1, S2 | Calgary landfill battery fires, ~40/yr | Fire-channel calibration |
    | S4 | $2.5B continental facility fire losses (2025) | Cost per fire (sensitivity range) |
    | S5 | 1-in-4 facility battery fires cause six-figure damage | Escalation bounds |
    | S6 | "Producers get off scot-free" (Fire Rover) | Problem framing |
    | S10 | EU battery passport 2027; small cells excluded | Tags channel; the gap argument |
    | S11 | Call2Recycle: PRO in ON/AB, programs in six provinces | Depot channel; legal addressee |
    | S12 | Tetra Tech waste-composition audits (BC) | Audit-channel realism |
    | S13 | Nova Scotia ~80% deposit-return rate | Deposit-intervention ceiling |
    | S14 | NS EPR live Dec 2025; producer-fee consultation | Legal pathway |
    | U1, U3, U4 | No municipal fire cost; legal authority; perpetual-care rate | **Open questions, labeled** |
            """
        )


def main() -> None:
    """Choose between the presenter path and the full analysis workbench."""
    st.set_page_config(page_title="Return to Sender Advisory", page_icon="🔋", layout="wide")
    apply_demo_styles()
    with st.sidebar:
        st.markdown("**RETURN TO SENDER ADVISORY**")
        st.caption("Battery-waste externality accounting / student prototype")
        experience = st.radio(
            "Experience",
            ["Guided demo", "Advanced analysis"],
            help="Guided demo is the deterministic ten-minute presentation. Advanced analysis "
                 "exposes CSV upload, every evidence channel, seeds, and timing controls.",
        )
        if experience == "Guided demo":
            st.info("Three fixed stages · Seed 0 · Synthetic data")

    if experience == "Guided demo":
        render_guided_app()
    else:
        render_advanced_app()


if runtime.exists():  # streamlit run / AppTest; bare imports skip rendering
    main()

