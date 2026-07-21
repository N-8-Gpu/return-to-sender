"""app.py -- Streamlit demo: toggle data channels, run the filter, watch the
posterior bands shrink and the externality invoice firm up.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import invoice as inv
from config import Config
from filter import run_filter
from simulator import simulate

st.set_page_config(page_title="Return to Sender", layout="wide")
st.title("Return to Sender: the externality invoice")

_DEFAULT_CFG = Config()  # only used for slider bounds/defaults below; seed doesn't affect these

with st.sidebar:
    st.header("Data channels")
    active_channels = {
        "audits": st.checkbox("Audits", value=True),
        "depots": st.checkbox("Depots", value=False),
        "fires": st.checkbox("Fires", value=False),
        "prices": st.checkbox("Prices", value=False),
        "mailbox": st.checkbox("Mailbox", value=False),
        "compactor": st.checkbox("Compactor", value=False),
        "tags": st.checkbox("Tags", value=False),
    }
    tau = st.slider("Tag adoption (tau)", 0.0, 0.3, 0.0, step=0.01, disabled=not active_channels["tags"])

    st.header("Deposit intervention")
    deposit_on = st.checkbox("Enable deposit-return intervention", value=False)
    t_deposit_input = st.slider(
        "Deposit start quarter", 0, _DEFAULT_CFG.geography.n_quarters - 1,
        value=_DEFAULT_CFG.phi_dynamics.t_deposit, disabled=not deposit_on,
    )

    st.header("Run settings")
    seed = st.number_input("Seed", min_value=0, value=0, step=1)
    run_clicked = st.button("Run", type="primary")

if not run_clicked:
    st.info("Set your channels and click **Run** in the sidebar to simulate a scenario and run the filter.")
    st.stop()

cfg = Config(seed=int(seed))
rng = np.random.default_rng(cfg.seed)
t_deposit = int(t_deposit_input) if deposit_on else None
tau_effective = tau if active_channels["tags"] else 0.0
scenario = simulate(cfg, rng, t_deposit=t_deposit, tau=tau_effective)
truth, obs = scenario["truth"], scenario["obs"]

try:
    results = run_filter(obs, active_channels, cfg, seed=cfg.seed)
except NotImplementedError:
    st.warning(
        "filter.py hasn't been implemented yet -- that part is written by hand. "
        "The rest of the app (simulator, invoice, plots) is ready and waiting for it."
    )
    st.stop()

t_axis = np.arange(cfg.geography.n_quarters)

st.subheader("Posterior leakage fraction vs. truth")
q_phi_bar = results["quantiles"]["phi_bar"]  # (T, 3): p05, p50, p95
fig1, ax1 = plt.subplots(figsize=(9, 4))
ax1.fill_between(t_axis, q_phi_bar[:, 0], q_phi_bar[:, 2], alpha=0.25, label="90% band")
ax1.plot(t_axis, q_phi_bar[:, 1], label="Posterior median")
ax1.plot(t_axis, truth["phi_bar"], linestyle="--", color="black", label="True phi_bar")
ax1.axvline(cfg.regime.t_shock, color="gray", linestyle=":", label="Market shock")
if t_deposit is not None:
    ax1.axvline(t_deposit, color="green", linestyle=":", label="Deposit starts")
ax1.set_xlabel("Quarter")
ax1.set_ylabel("Leakage fraction (phi_bar)")
ax1.legend(loc="upper right", fontsize=8)
st.pyplot(fig1)

st.subheader("Filter health: effective sample size")
fig2, ax2 = plt.subplots(figsize=(9, 1.5))
ax2.plot(t_axis, results["n_eff"])
ax2.set_xlabel("Quarter")
ax2.set_ylabel("N_eff")
st.pyplot(fig2)

st.subheader("The externality invoice")
posterior = inv.compute_cost_posterior(results, cfg)
modulation = inv.modulation_table(posterior["total_median"], posterior["total_leaked_tonnage"], cfg)
fig3 = inv.render_invoice(posterior, modulation, seed=cfg.seed)
st.pyplot(fig3)

with st.expander("4-beat demo script"):
    st.markdown(
        """
1. **Start with only audits on.** Point at the wide 90% band -- we're basically guessing.
2. **Turn on mailbox, then compactor.** Watch the band visibly narrow around the
   true phi_bar -- more channels, sharper estimate, same underlying truth.
3. **Enable the deposit intervention.** The true phi_bar bends down a few quarters
   after the toggle; the posterior catches up.
4. **Point at the invoice.** Every channel you just turned on didn't lower the bill --
   it made the number defensible. The billable floor is what you can charge with
   confidence, today, with the data you already have.
        """
    )

with st.expander("Mass-balance check (by brand, simulator-only)"):
    tally = obs["brand_tally"]
    fig4, ax4 = plt.subplots(figsize=(9, 3))
    for i, name in enumerate(cfg.brand.brand_names):
        ax4.plot(t_axis, tally[:, i], label=name)
    ax4.set_xlabel("Quarter")
    ax4.set_ylabel("Proper-return tally")
    ax4.legend(fontsize=8)
    st.pyplot(fig4)
