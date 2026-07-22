# Return to Sender: prototype

## What this is
SHAD 2026 design-challenge prototype. A synthetic-data demonstration of a closed-loop system that estimates the hidden downstream costs of lithium-ion battery waste (fires, landfill space, destroyed material value) using a Sequential Monte Carlo (particle) filter, and renders the result as an "externality invoice." The demo centerpiece: toggling data channels on and off and watching the posterior uncertainty bands shrink, which argues that every proposed program makes the invoice fairer.

Built by Nolan (Grade 10, SHAD Calgary) with Claude Code. Nolan previously built a Kalman filter on a Raspberry Pi for radon monitoring (CWSF project), so he knows state-space estimation; explain non-obvious code decisions in one-line comments, and prefer clarity over cleverness.

## Division of labor
- **Claude Code builds:** `config.py`, `simulator.py`, `app.py`, `invoice.py`, `filter.py`, `tests/`, plotting, and all glue.
- **RETIRED (2026-07-21):** `filter.py` was originally reserved for Nolan to write by hand, so "I wrote the estimator" stayed true on stage at SHAD 2026. Nolan explicitly lifted this restriction — the project's use changed and it's no longer presented as a judged authorship claim — so Claude Code implemented `filter.py` in full. If the project ever returns to being presented as Nolan's own estimator work, flag that the implementation is Claude-written.

## File layout
```
returntosender/
  config.py        # every parameter in one place (dataclasses); geography/product = config
  simulator.py     # ground truth + all synthetic observation channels
  filter.py        # particle filter core (Claude-written; see Division of labor)
  invoice.py       # cost computation + rendered invoice figure
  app.py           # Streamlit demo
  tests/test_filter.py
  requirements.txt # pinned versions
```

## The model (implement exactly this in simulator.py)
Time step = one quarter, T = 40 (ten years). Two design classes c ∈ {embedded, removable}, class shares fixed in config (default 0.6 / 0.4). RNG: `numpy.random.default_rng(seed)` everywhere; seed lives in config and the app sidebar.

### Hidden states
- `X_t` : end-of-life tonnage (keep in log space internally). Anchored to sales through a lifespan kernel:
  `X_t = a*X_{t-1} + (1-a)*log(sum_k kappa_k * S_{t-k}) + Normal(0, q_X)`
  where `S_t` is the disclosed sales series (a config input, e.g. gently growing), and `kappa` is a discretized lognormal lifespan kernel (mean ~8 quarters).
- `phi_t^c` : leakage fraction per class, random walk on logit scale:
  `logit(phi_t^c) = logit(phi_{t-1}^c) + beta*u_t + Normal(0, q_phi)`
  `u_t` = deposit-intervention indicator (0 until `t_deposit`, then 1). beta < 0 (the deposit lowers leakage). Class split of X: `X_t^c = w_c * X_t`.
- `r_t` : market regime, 2-state Markov chain (p01, p10 in config). Force a shift to regime 1 at `t_shock = 12` in the default scenario (National Sword event).
- `s_t` : recyclate-virgin price spread, regime-switching AR(1):
  `s_t = mu[r_t] + rho[r_t]*(s_{t-1} - mu[r_t]) + Normal(0, q_s[r_t])`
- `L_t` : log fire intensity per leaked tonne, slow random walk `L_t = L_{t-1} + Normal(0, q_L)`.

### Observation channels (all quarterly in v1; each is a function of truth + noise)
1. **audits**: `k_t ~ Binomial(n_audit, d * phi_bar_t)` per class (auditors can see embedded vs removable), n_audit = 200 bags, d = 0.7 detection.
2. **depots**: `D_t ~ LogNormal(log((1 - phi_bar_t) * X_t), sigma_D)`.
3. **fires**: `F_t ~ Poisson(exp(L_t) * (1.0*phi^emb*X^emb + 0.4*phi^rem*X^rem))` (embedded more fire-prone). Calibrate config so the default scenario yields roughly 10 fires per quarter (~40/yr, the Calgary anchor).
4. **prices**: `P_t ~ Normal(s_t, sigma_P)`.
5. **mailbox**: `m_t ~ Poisson(c_mail * (1 - phi_bar_t) * X_t * uptake_t)`, active only after `t_mailbox = 8`, uptake ramping 0 → 1 over 4 quarters.
6. **compactor**: `H_t ~ Poisson(alpha_comp * phi_bar_t * X_t)` with alpha_comp large (high-count, information-rich channel; stands in for daily sensor data).
7. **plant brand tallies** (simulator only, feeds the optional mass-balance chart, NOT the filter in v1): 3 synthetic brands with sales shares and distinct phi^b; tally `~ Multinomial` over brands among proper returns.
8. **tags**: stylized. Tag-adoption slider tau ∈ [0, 0.3]. When tau > 0, emit a direct observation `z_t ~ Normal(phi_bar_t, sigma_tag(tau))` with `sigma_tag = 0.12*(1 - tau) + 0.01`. Document clearly in a comment that this is a stylized stand-in for per-unit passport data.

`phi_bar_t` = class-share-weighted mean leakage. The simulator returns a dict of truth arrays and a dict of observation arrays, each keyed by channel name, with NaN where a channel is off or not yet active.

### The invoice (invoice.py)
Per quarter: `C_t = phi_bar_t * X_t * (u_LF + exp(L_t)*u_F + max(s_t - u_sort, 0))`, decomposed into the three line items (landfill space and perpetual care; expected fire cost; destroyed material value). Given the filter's weighted particles, compute posterior median, 90% band, and the 5th-percentile "billable floor" of cumulative cost. Render a styled invoice figure: addressed to "Consumer Lithium-Ion Battery Producers, via stewardship organization"; three line items; a small modulation table (embedded rate vs removable rate, ratio from config); billable floor highlighted. Matplotlib is fine; make it look like a bill, not a chart.

## filter.py contract
```python
def systematic_resample(weights, rng) -> indices
def pf_step(particles, weights, obs_t, active_channels,
            params, rng) -> (particles, weights, n_eff)
def run_filter(observations, active_channels, params,
               n_particles=5000, seed=0) -> results
```
`results` holds per-quarter weighted quantiles for each state, N_eff trace, and the particle cloud needed by invoice.py. Predict via simulator dynamics, weight by product of active-channel likelihoods (scipy.stats pmf/pdf), normalize, resample when N_eff < N/2. Likelihood helper functions live in simulator.py (shared with the simulator) so the math has one home.

## app.py (Streamlit)
Sidebar: checkboxes per channel (audits, depots, fires, prices, mailbox, compactor, tags with tau slider), deposit-intervention toggle + quarter slider, seed number input, Run button. Main page, top to bottom: (1) the shrinking-bands chart: true phi_bar vs posterior median + 90% band, vertical markers at the shock and intervention quarters; (2) N_eff sparkline (filter health, judges like diagnostics); (3) the invoice figure; (4) expander with the 4-beat demo script; (5) optional mass-balance brand chart. Everything reruns in under ~2 seconds on a laptop (5000 particles x 40 quarters is milliseconds; keep it vectorized).

## Tests (tests/test_filter.py) the filter core must pass
1. `test_resample_preserves_mean`: resampling a known weighted cloud approximately preserves the weighted mean.
2. `test_tracks_constant_truth`: with truth held constant and all channels on, posterior median bias on phi_bar is small.
3. `test_regime_detected`: posterior mean of s_t moves toward the new regime mean within 3 quarters of t_shock.
4. `test_more_channels_narrower`: mean 90% band width on phi_bar strictly narrows (averaged over 5 seeds) going audits-only → +mailbox → +compactor → +tags(0.2).
5. `test_coverage` (mark slow): over 20 seeds, the 90% band covers true phi_bar in roughly 85 to 95% of quarter-instances.

## Conventions and guardrails
- Pinned `requirements.txt` (numpy, scipy, matplotlib, streamlit, pytest). No other dependencies. No network access, no real data downloads.
- Seeded and reproducible end to end. Type hints. Small pure functions. Plain-language docstrings (judges will read this code).
- DO NOT build: hardware integration, web scraping, databases, multi-page apps, the full per-brand filter, or any real mailbox/sensor program. Those exist in the pitch, not the prototype (spec section 8).
- Commit style: small commits with plain messages after each working step.

## Run
```
pip install -r requirements.txt
pytest -m "not slow"
streamlit run app.py
```

## The four verbs (filter.py's per-quarter loop)
```
for each quarter t:
    GUESS:  push every particle through the dynamics (sample noise, flip regime coin)
    WEIGHT: w_i *= product over active channels of likelihood(obs_t | particle_i)
    NORM:   w /= sum(w);  N_eff = 1 / sum(w**2)
    CULL:   if N_eff < N/2: systematic resample, reset w to 1/N
    RECORD: weighted quantiles of each state
```
Same predict-correct heartbeat as a Kalman filter; sampling replaces algebra.
