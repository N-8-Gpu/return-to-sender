"""config.py -- every model parameter lives here, grouped by what it describes.

Citation convention (see SOURCES.md): a comment tag like [S4] points at a verified
source entry; [U1] points at an unverified one (use only with a sensitivity range);
ASSUMPTION marks a plain modeling choice with no source, given a plausible range.
Numbers with none of these are literal values fixed by the project brief itself
(CLAUDE.md), not independent claims about the world.

Geography/product scope for v1: a single mid-size Canadian city (Calgary-anchored
fire calibration, [S1][S2]), two battery design classes, forty quarterly steps.
"""

from dataclasses import dataclass, field
import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Logistic function; inverse of logit."""
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray | float) -> np.ndarray | float:
    """Log-odds; inverse of sigmoid."""
    return np.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Geography / product: sales series and the lifespan kernel that turns sales
# into end-of-life tonnage.
# ---------------------------------------------------------------------------

@dataclass
class GeographyConfig:
    """Product scope: design-class split and the disclosed sales series."""
    class_names: tuple[str, str] = ("embedded", "removable")
    class_share: tuple[float, float] = (0.6, 0.4)  # ASSUMPTION: no public per-class split; ratio fixed by brief
    n_quarters: int = 40  # T = ten years of quarters (brief)
    sales_t0: float = 200.0  # tonnes of battery-containing product sold, quarter 0; ASSUMPTION scale
    sales_growth: float = 0.015  # gentle per-quarter growth; ASSUMPTION


def sales_series(geo: GeographyConfig) -> np.ndarray:
    """Disclosed sales S_t: a gently growing, deterministic series (a 'known' input)."""
    t = np.arange(geo.n_quarters)
    return geo.sales_t0 * (1.0 + geo.sales_growth) ** t


@dataclass
class LifespanKernelConfig:
    """Discretized lognormal lifespan kernel: how long batteries sit before end-of-life."""
    mean_quarters: float = 8.0  # ~2 years, brief
    sigma: float = 0.5  # lognormal shape; ASSUMPTION (moderate spread around the mean)
    n_lags: int = 24  # truncate the kernel tail at 24 quarters back


def lifespan_kernel(cfg: LifespanKernelConfig) -> np.ndarray:
    """Return kappa[0..n_lags-1]: a normalized lognormal pmf over lag k = 1..n_lags.

    Lag 0 (same-quarter retirement) is excluded -- nothing dies the instant it ships.
    """
    k = np.arange(1, cfg.n_lags + 1)
    mu = np.log(cfg.mean_quarters) - 0.5 * cfg.sigma ** 2  # mean-match the lognormal
    pdf = np.exp(-((np.log(k) - mu) ** 2) / (2 * cfg.sigma ** 2)) / (k * cfg.sigma * np.sqrt(2 * np.pi))
    return pdf / pdf.sum()


# ---------------------------------------------------------------------------
# Hidden-state dynamics
# ---------------------------------------------------------------------------

@dataclass
class XDynamicsConfig:
    """End-of-life tonnage X_t: anchored to the lifespan-kernel-weighted sales history."""
    a: float = 0.7  # persistence weight vs. the sales anchor; ASSUMPTION (slow-moving state)
    q_x: float = 0.02  # process noise variance, log space; ASSUMPTION


@dataclass
class PhiDynamicsConfig:
    """Leakage fraction phi_t^c per class: random walk on the logit scale, nudged by the deposit."""
    phi0: tuple[float, float] = (0.40, 0.25)  # (embedded, removable) baseline leakage; ASSUMPTION
    # Embedded > removable: harder to remove for proper disposal (design assumption, no direct source).
    beta: float = -0.09  # per-quarter logit shift once the deposit is active; ASSUMPTION
    # Calibrated so ~12 quarters of an active deposit pulls phi_bar down toward the
    # Nova Scotia ~80%-capture ceiling observed in practice [S13], not an instant jump.
    q_phi: float = 0.02  # process noise sd, logit scale; ASSUMPTION (kept small: leakage drifts slowly)
    t_deposit: int | None = 20  # quarter the deposit intervention switches on; None = never active
    # Single source of truth: simulator.py reads this field (not a separate function
    # argument) so the filter can replicate the exact same intervention timing from
    # this same Config instance. Set it (and tag.tau below) before calling simulate().


@dataclass
class RegimeConfig:
    """Two-state Markov chain for market regime r_t, forced to flip at the Sword shock."""
    p01: float = 0.02  # spontaneous prob of regime 0 -> 1 per quarter; ASSUMPTION
    p10: float = 0.05  # spontaneous prob of regime 1 -> 0 per quarter; ASSUMPTION
    t_shock: int = 12  # forced shift to regime 1 (National Sword event), per brief


@dataclass
class PriceSpreadConfig:
    """Recyclate-virgin price spread s_t: regime-switching AR(1), $ per tonne."""
    mu: tuple[float, float] = (100.0, 180.0)  # per-regime mean spread; ASSUMPTION
    rho: tuple[float, float] = (0.7, 0.7)  # per-regime AR(1) persistence; ASSUMPTION
    q_s: tuple[float, float] = (15.0, 15.0)  # per-regime process noise sd; ASSUMPTION


@dataclass
class FireIntensityConfig:
    """Log fire intensity per leaked tonne, L_t: slow random walk."""
    l0: float | None = None  # calibrated in Config.__post_init__ if left None
    q_l: float = 0.03  # process noise sd; ASSUMPTION
    target_fires_per_quarter: float = 10.0  # Calgary anchor, ~40/yr [S1][S2]
    fire_mult: tuple[float, float] = (1.0, 0.4)  # (embedded, removable) relative fire-proneness, per brief


# ---------------------------------------------------------------------------
# Observation channels
# ---------------------------------------------------------------------------

@dataclass
class AuditConfig:
    """Per-class audit sampling: bags checked, detection rate."""
    n_audit: int = 200  # bags sampled per class per quarter, per brief
    detection: float = 0.7  # probability an auditor correctly flags a leaked bag, per brief


@dataclass
class DepotConfig:
    """Depot-reported proper-return tonnage, lognormal noise."""
    sigma_d: float = 0.15  # log-space observation noise sd; ASSUMPTION


@dataclass
class PriceObsConfig:
    """Observed price spread, additive Gaussian noise."""
    sigma_p: float = 10.0  # $/tonne observation noise sd; ASSUMPTION


@dataclass
class MailboxConfig:
    """Stylized mailback-program counts: active only after t_mailbox, uptake ramps in."""
    c_mail: float = 0.05  # rate-scaling constant; ASSUMPTION, stylized stand-in for a pilot program
    t_mailbox: int = 8  # quarter the program switches on, per brief
    uptake_ramp: int = 4  # quarters for uptake to ramp 0 -> 1, per brief


@dataclass
class CompactorConfig:
    """Compactor sensor counts: high-count, information-rich channel (stands in for daily sensor data)."""
    alpha_comp: float = 2.0  # rate-scaling constant, deliberately large; ASSUMPTION


@dataclass
class BrandConfig:
    """Synthetic per-brand sales shares and leakage, for the optional mass-balance chart only.

    Not fed to the filter in v1 (brief, section on plant brand tallies).
    """
    brand_names: tuple[str, str, str] = ("Brand A", "Brand B", "Brand C")
    sales_share: tuple[float, float, float] = (0.5, 0.3, 0.2)  # ASSUMPTION
    phi_brand: tuple[float, float, float] = (0.30, 0.40, 0.20)  # ASSUMPTION


@dataclass
class TagConfig:
    """Stylized battery-passport / tag-adoption channel.

    tau in [0, 0.3] is a stand-in for per-unit passport rollout share, not the EU
    battery-passport mandate itself (that covers EV/LMT/industrial cells >2 kWh,
    not small consumer cells, per [S10] -- this channel is a deliberate stylization,
    documented here so nobody mistakes it for a real passport data feed).
    """
    tau: float = 0.0  # off by default; app exposes a 0-0.3 slider
    sigma_tag_floor: float = 0.01  # noise floor as tau -> 0.3, exact formula in simulator.py
    sigma_tag_scale: float = 0.12  # noise scale at tau = 0


# ---------------------------------------------------------------------------
# Invoice unit costs
# ---------------------------------------------------------------------------

@dataclass
class InvoiceConfig:
    """Per-tonne unit costs turning leaked tonnage into a dollar invoice."""
    u_lf: float = 150.0  # $/tonne landfill perpetual-care cost; ASSUMPTION, no PSAB figure found [U4]
    u_f: float = 250_000.0  # $ cost per fire; ASSUMPTION anchored to continental data [S4], see [U1]
    # S4: ~$2.5B / 448 incidents (~100 catastrophic) is a continental, facility-fire figure,
    # not a Calgary per-incident cost -- no such figure exists [U1]. u_f is set well below
    # the naive average to reflect that most incidents are minor and a few are catastrophic;
    # sweep u_f over roughly $50k-$2M for sensitivity, not a point estimate.
    u_sort: float = 80.0  # $/tonne sorting/material-recovery cost baseline; ASSUMPTION
    # kept below price_spread.mu (100, 180) so the destroyed-material-value line item
    # isn't structurally zero -- it should sometimes bite, not always floor at 0
    modulation_ratio: float = 1.4  # embedded stewardship fee / removable fee; ASSUMPTION


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Every model parameter, grouped by sub-config. Pass one instance around everywhere."""
    seed: int = 0
    geography: GeographyConfig = field(default_factory=GeographyConfig)
    lifespan: LifespanKernelConfig = field(default_factory=LifespanKernelConfig)
    x_dynamics: XDynamicsConfig = field(default_factory=XDynamicsConfig)
    phi_dynamics: PhiDynamicsConfig = field(default_factory=PhiDynamicsConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    price_spread: PriceSpreadConfig = field(default_factory=PriceSpreadConfig)
    fire_intensity: FireIntensityConfig = field(default_factory=FireIntensityConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    depot: DepotConfig = field(default_factory=DepotConfig)
    price_obs: PriceObsConfig = field(default_factory=PriceObsConfig)
    mailbox: MailboxConfig = field(default_factory=MailboxConfig)
    compactor: CompactorConfig = field(default_factory=CompactorConfig)
    brand: BrandConfig = field(default_factory=BrandConfig)
    tag: TagConfig = field(default_factory=TagConfig)
    invoice: InvoiceConfig = field(default_factory=InvoiceConfig)

    def __post_init__(self) -> None:
        if self.fire_intensity.l0 is None:
            self.fire_intensity.l0 = _calibrate_log_fire_intensity(self)


def _calibrate_log_fire_intensity(cfg: Config) -> float:
    """Solve L0 so quarter-0 expected fire count hits the Calgary anchor [S1][S2].

    Uses the steady-state approximation X_0 ~= sales_t0 (kappa sums to one and sales
    is nearly flat over the kernel's support at t=0), split by class share and baseline
    leakage, weighted by each class's relative fire-proneness.
    """
    x0 = cfg.geography.sales_t0
    emb_share, rem_share = cfg.geography.class_share
    phi_emb0, phi_rem0 = cfg.phi_dynamics.phi0
    mult_emb, mult_rem = cfg.fire_intensity.fire_mult
    leaked_weighted = (
        mult_emb * phi_emb0 * emb_share * x0
        + mult_rem * phi_rem0 * rem_share * x0
    )
    return float(np.log(cfg.fire_intensity.target_fires_per_quarter / leaked_weighted))
