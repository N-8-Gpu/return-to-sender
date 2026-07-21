"""simulator.py -- ground truth generator, synthetic observation channels, and the
shared dynamics/likelihood math that filter.py's particle filter calls into.

Convention shared by every hidden state: the config value that looks like an
"initial condition" (the lifespan anchor at t=0, phi0, l0, mu[r_0]) is treated as
a virtual t=-1 value. One noisy recursion step turns it into the t=0 value, so
every quarter -- including the first -- goes through the same predict equation.

Design split: `simulate_truth`/`simulate_observations` generate one concrete
scenario with scalar per-quarter draws (used to build the demo and the tests).
The `predict_*` and `*_loglik` functions below are vectorized over a particle
array and are what filter.py's pf_step calls each quarter; they implement the
exact same equations, just over N particles instead of one ground truth.

One deliberate asymmetry: `predict_regime` (used by the filter) does NOT know
about cfg.regime.t_shock. Only `simulate_truth`'s ground truth is forced to
shock at that quarter -- the filter has to infer the regime shift from evidence
(mainly the price channel), same as it would for any real, undisclosed event.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from config import Config, lifespan_kernel, logit, sigmoid

# ---------------------------------------------------------------------------
# Sales history and the lifespan-kernel anchor
# ---------------------------------------------------------------------------

def _extended_sales(cfg: Config, n_lags: int) -> np.ndarray:
    """Sales series extended n_lags quarters into the past (same growth trend),
    so the lifespan kernel has history to look back on even at quarter 0.
    Index t_ext = n_lags + t maps to calendar quarter t (t can be negative).
    """
    t = np.arange(-n_lags, cfg.geography.n_quarters)
    return cfg.geography.sales_t0 * (1.0 + cfg.geography.sales_growth) ** t


def make_x_anchor_series(cfg: Config) -> np.ndarray:
    """Precompute anchor_t = log(sum_k kappa_k * S_{t-k}) for every quarter t.

    This does not depend on any particle, so both simulate_truth and the filter's
    predict step share one precomputed array instead of recomputing the kernel sum.
    """
    kappa = lifespan_kernel(cfg.lifespan)
    n_lags = len(kappa)
    sales_ext = _extended_sales(cfg, n_lags)
    offsets = n_lags - np.arange(1, n_lags + 1)  # indices for k = 1..n_lags, relative to t
    T = cfg.geography.n_quarters
    anchors = np.empty(T)
    for t in range(T):
        anchors[t] = np.log(np.dot(kappa, sales_ext[offsets + t]))
    return anchors


def mailbox_uptake(t: int, cfg: Config) -> float:
    """Uptake fraction for the mailback channel at quarter t: 0 before t_mailbox,
    ramping linearly to 1 over cfg.mailbox.uptake_ramp quarters."""
    if t < cfg.mailbox.t_mailbox:
        return 0.0
    return float(np.clip((t - cfg.mailbox.t_mailbox) / cfg.mailbox.uptake_ramp, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Ground truth: one concrete scenario, scalar per-quarter draws
# ---------------------------------------------------------------------------

def simulate_truth(cfg: Config, rng: np.random.Generator, t_deposit: int | None) -> dict[str, np.ndarray]:
    """Simulate the five hidden states over cfg.geography.n_quarters quarters.

    t_deposit: quarter the deposit intervention switches on (u_t becomes 1 from
    then on), or None if it never does.

    Returns arrays keyed by: X, X_emb, X_rem, phi_emb, phi_rem, phi_bar, r, s, L, u.
    """
    T = cfg.geography.n_quarters
    anchors = make_x_anchor_series(cfg)
    w_emb, w_rem = cfg.geography.class_share

    log_x = np.empty(T)
    logit_phi_emb = np.empty(T)
    logit_phi_rem = np.empty(T)
    r = np.empty(T, dtype=int)
    s = np.empty(T)
    L = np.empty(T)
    u = np.empty(T)

    a = cfg.x_dynamics.a
    q_x_sd = np.sqrt(cfg.x_dynamics.q_x)
    beta = cfg.phi_dynamics.beta
    q_phi_sd = np.sqrt(cfg.phi_dynamics.q_phi)
    q_l_sd = np.sqrt(cfg.fire_intensity.q_l)
    p01, p10 = cfg.regime.p01, cfg.regime.p10
    mu_s, rho_s, q_s = cfg.price_spread.mu, cfg.price_spread.rho, cfg.price_spread.q_s

    log_x_prev = anchors[0]  # X_{-1} := the t=0 anchor (steady-state assumption)
    logit_phi_emb_prev, logit_phi_rem_prev = logit(np.array(cfg.phi_dynamics.phi0))
    l_prev = cfg.fire_intensity.l0
    r_prev = 0  # regime starts at 0 unless t_shock == 0
    s_prev = mu_s[0]  # s_{-1} := the regime-0 mean

    for t in range(T):
        u_t = 1.0 if (t_deposit is not None and t >= t_deposit) else 0.0
        u[t] = u_t

        log_x[t] = a * log_x_prev + (1 - a) * anchors[t] + rng.normal(0.0, q_x_sd)
        log_x_prev = log_x[t]

        logit_phi_emb[t] = logit_phi_emb_prev + beta * u_t + rng.normal(0.0, q_phi_sd)
        logit_phi_rem[t] = logit_phi_rem_prev + beta * u_t + rng.normal(0.0, q_phi_sd)
        logit_phi_emb_prev = logit_phi_emb[t]
        logit_phi_rem_prev = logit_phi_rem[t]

        if t == cfg.regime.t_shock:
            r_t = 1  # forced National Sword shock; ground truth only, filter must infer this
        elif t == 0:
            r_t = r_prev
        else:
            draw = rng.random()
            if r_prev == 0:
                r_t = 1 if draw < p01 else 0
            else:
                r_t = 0 if draw < p10 else 1
        r[t] = r_t
        r_prev = r_t

        mu_r, rho_r, q_r = mu_s[r_t], rho_s[r_t], q_s[r_t]
        s[t] = mu_r + rho_r * (s_prev - mu_r) + rng.normal(0.0, np.sqrt(q_r))
        s_prev = s[t]

        L[t] = l_prev + rng.normal(0.0, q_l_sd)
        l_prev = L[t]

    phi_emb = sigmoid(logit_phi_emb)
    phi_rem = sigmoid(logit_phi_rem)
    phi_bar = w_emb * phi_emb + w_rem * phi_rem
    X = np.exp(log_x)

    return {
        "X": X,
        "X_emb": w_emb * X,
        "X_rem": w_rem * X,
        "phi_emb": phi_emb,
        "phi_rem": phi_rem,
        "phi_bar": phi_bar,
        "r": r,
        "s": s,
        "L": L,
        "u": u,
    }


def simulate_observations(cfg: Config, rng: np.random.Generator, truth: dict[str, np.ndarray], tau: float) -> dict[str, np.ndarray]:
    """Draw every observation channel from the ground truth. Each channel is a
    function of truth + noise; NaN marks a channel that is off or not yet active.
    """
    T = cfg.geography.n_quarters
    X, X_emb, X_rem = truth["X"], truth["X_emb"], truth["X_rem"]
    phi_emb, phi_rem, phi_bar = truth["phi_emb"], truth["phi_rem"], truth["phi_bar"]
    s, L = truth["s"], truth["L"]

    # 1. audits: auditors can tell embedded from removable, so each class is audited separately.
    p_emb = np.clip(cfg.audit.detection * phi_emb, 0.0, 1.0)
    p_rem = np.clip(cfg.audit.detection * phi_rem, 0.0, 1.0)
    audits_emb = rng.binomial(cfg.audit.n_audit, p_emb).astype(float)
    audits_rem = rng.binomial(cfg.audit.n_audit, p_rem).astype(float)

    # 2. depots: lognormal around the proper-return tonnage.
    depots = rng.lognormal(np.log(np.maximum((1 - phi_bar) * X, 1e-9)), cfg.depot.sigma_d)

    # 3. fires: embedded batteries are more fire-prone than removable ones.
    mult_emb, mult_rem = cfg.fire_intensity.fire_mult
    fire_rate = np.exp(L) * (mult_emb * phi_emb * X_emb + mult_rem * phi_rem * X_rem)
    fires = rng.poisson(fire_rate).astype(float)

    # 4. prices: noisy read of the true spread.
    prices = rng.normal(s, cfg.price_obs.sigma_p)

    # 5. mailbox: stylized mailback pilot, off before t_mailbox, uptake ramps in after.
    uptake = np.array([mailbox_uptake(t, cfg) for t in range(T)])
    mailbox_rate = cfg.mailbox.c_mail * (1 - phi_bar) * X * uptake
    mailbox = rng.poisson(np.maximum(mailbox_rate, 0.0)).astype(float)
    mailbox[np.arange(T) < cfg.mailbox.t_mailbox] = np.nan

    # 6. compactor: high-count, information-rich stand-in for daily sensor data.
    compactor = rng.poisson(cfg.compactor.alpha_comp * phi_bar * X).astype(float)

    # 7. plant brand tallies: simulator-only, feeds the optional mass-balance chart,
    #    NOT the filter in v1. Proper-return volume splits across brands by sales
    #    share weighted by each brand's own (fixed, not time-varying) leakage rate.
    shares = np.array(cfg.brand.sales_share)
    phi_brand = np.array(cfg.brand.phi_brand)
    proper_weight = shares * (1 - phi_brand)
    p_brand = proper_weight / proper_weight.sum()
    total_proper = np.maximum((1 - phi_bar) * X, 0.0).astype(int)
    brand_tally = np.array([rng.multinomial(n, p_brand) for n in total_proper])

    # 8. tags: stylized stand-in for per-unit passport data (see config.TagConfig).
    #    Off entirely (NaN) unless tau > 0.
    if tau > 0:
        sigma_tag = cfg.tag.sigma_tag_scale * (1 - tau) + cfg.tag.sigma_tag_floor
        tags = rng.normal(phi_bar, sigma_tag)
    else:
        tags = np.full(T, np.nan)

    return {
        "audits_emb": audits_emb,
        "audits_rem": audits_rem,
        "depots": depots,
        "fires": fires,
        "prices": prices,
        "mailbox": mailbox,
        "compactor": compactor,
        "brand_tally": brand_tally,
        "tags": tags,
    }


def simulate(cfg: Config, rng: np.random.Generator, t_deposit: int | None, tau: float) -> dict[str, dict[str, np.ndarray]]:
    """Run the full synthetic scenario: hidden states, then every observation channel.
    Returns {'truth': {...}, 'obs': {...}}.
    """
    truth = simulate_truth(cfg, rng, t_deposit)
    obs = simulate_observations(cfg, rng, truth, tau)
    return {"truth": truth, "obs": obs}


# ---------------------------------------------------------------------------
# Vectorized predict-step dynamics, for filter.py's pf_step (one call per
# quarter, arrays are shaped (n_particles,))
# ---------------------------------------------------------------------------

def predict_x(log_x: np.ndarray, anchor_t: float, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """One predict step for log end-of-life tonnage, vectorized over particles."""
    noise = rng.normal(0.0, np.sqrt(cfg.x_dynamics.q_x), size=log_x.shape)
    return cfg.x_dynamics.a * log_x + (1 - cfg.x_dynamics.a) * anchor_t + noise


def predict_phi_logit(logit_phi: np.ndarray, u_t: float, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """One predict step for a class's leakage logit, vectorized over particles."""
    noise = rng.normal(0.0, np.sqrt(cfg.phi_dynamics.q_phi), size=logit_phi.shape)
    return logit_phi + cfg.phi_dynamics.beta * u_t + noise


def predict_regime(r: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """One predict step for the two-state regime chain, ordinary Markov transitions only.

    Deliberately does NOT know about cfg.regime.t_shock: particles must infer the
    National Sword shift from evidence (mainly the price channel), not be told about it.
    """
    draw = rng.random(size=r.shape)
    flip_to_1 = (r == 0) & (draw < cfg.regime.p01)
    flip_to_0 = (r == 1) & (draw < cfg.regime.p10)
    return np.where(flip_to_1, 1, np.where(flip_to_0, 0, r))


def predict_price_spread(s: np.ndarray, r: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """One predict step for the price spread, vectorized over particles; r selects each particle's regime."""
    mu = np.asarray(cfg.price_spread.mu)[r]
    rho = np.asarray(cfg.price_spread.rho)[r]
    q = np.asarray(cfg.price_spread.q_s)[r]
    noise = rng.normal(0.0, np.sqrt(q))
    return mu + rho * (s - mu) + noise


def predict_fire_intensity(L: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """One predict step for log fire intensity, vectorized over particles."""
    noise = rng.normal(0.0, np.sqrt(cfg.fire_intensity.q_l), size=L.shape)
    return L + noise


# ---------------------------------------------------------------------------
# Vectorized log-likelihoods, for filter.py's weight step. Each takes a scalar
# observation and particle-array latent state(s), and returns a per-particle
# log-likelihood array. NaN observations (channel off/inactive) contribute 0.
# ---------------------------------------------------------------------------

def audit_loglik(k: float, phi: np.ndarray, cfg: Config) -> np.ndarray:
    """Binomial log-likelihood of one class's audit count given per-particle leakage."""
    if np.isnan(k):
        return np.zeros_like(phi)
    p = np.clip(cfg.audit.detection * phi, 1e-9, 1 - 1e-9)
    return stats.binom.logpmf(k, cfg.audit.n_audit, p)


def depot_loglik(d: float, phi_bar: np.ndarray, x: np.ndarray, cfg: Config) -> np.ndarray:
    """Lognormal log-likelihood of the depot-reported proper-return tonnage."""
    if np.isnan(d):
        return np.zeros_like(phi_bar)
    mean = np.log(np.maximum((1 - phi_bar) * x, 1e-9))
    return stats.norm.logpdf(np.log(d), loc=mean, scale=cfg.depot.sigma_d)


def fire_loglik(f: float, phi_emb: np.ndarray, phi_rem: np.ndarray, x_emb: np.ndarray, x_rem: np.ndarray,
                l: np.ndarray, cfg: Config) -> np.ndarray:
    """Poisson log-likelihood of the quarterly fire count."""
    if np.isnan(f):
        return np.zeros_like(phi_emb)
    mult_emb, mult_rem = cfg.fire_intensity.fire_mult
    rate = np.exp(l) * (mult_emb * phi_emb * x_emb + mult_rem * phi_rem * x_rem)
    return stats.poisson.logpmf(f, np.maximum(rate, 1e-12))


def price_loglik(p: float, s: np.ndarray, cfg: Config) -> np.ndarray:
    """Gaussian log-likelihood of the observed price spread."""
    if np.isnan(p):
        return np.zeros_like(s)
    return stats.norm.logpdf(p, loc=s, scale=cfg.price_obs.sigma_p)


def mailbox_loglik(m: float, phi_bar: np.ndarray, x: np.ndarray, uptake: float, cfg: Config) -> np.ndarray:
    """Poisson log-likelihood of the mailback count; caller passes uptake = mailbox_uptake(t, cfg)."""
    if np.isnan(m):
        return np.zeros_like(phi_bar)
    rate = cfg.mailbox.c_mail * (1 - phi_bar) * x * uptake
    return stats.poisson.logpmf(m, np.maximum(rate, 1e-12))


def compactor_loglik(h: float, phi_bar: np.ndarray, x: np.ndarray, cfg: Config) -> np.ndarray:
    """Poisson log-likelihood of the compactor sensor count."""
    if np.isnan(h):
        return np.zeros_like(phi_bar)
    rate = cfg.compactor.alpha_comp * phi_bar * x
    return stats.poisson.logpmf(h, np.maximum(rate, 1e-12))


def tag_loglik(z: float, phi_bar: np.ndarray, tau: float, cfg: Config) -> np.ndarray:
    """Gaussian log-likelihood of the stylized tag/passport reading; zero contribution if tau <= 0."""
    if np.isnan(z) or tau <= 0:
        return np.zeros_like(phi_bar)
    sigma = cfg.tag.sigma_tag_scale * (1 - tau) + cfg.tag.sigma_tag_floor
    return stats.norm.logpdf(z, loc=phi_bar, scale=sigma)
