"""tests/test_csv_io.py -- the real-data upload path.

The round-trip test is the important one: a CSV produced by the template
function must parse back into an obs dict the filter can run on, giving the
same posterior it would give on the equivalent synthetic dict.
"""

import numpy as np
import pytest

from config import Config
from filter import run_filter
from simulator import OBS_CHANNELS, observations_csv_template, parse_observations_csv


def test_template_round_trips_through_parser():
    cfg = Config()
    csv_text = observations_csv_template(cfg)
    obs, T, sales = parse_observations_csv(csv_text)
    assert T == cfg.geography.n_quarters
    assert set(obs) == set(OBS_CHANNELS)
    assert sales is not None and len(sales) == T and np.all(sales > 0)
    # mailbox is inactive for its first 8 quarters and tags are off (tau=0):
    assert np.isnan(obs["mailbox"][:8]).all()
    assert np.isnan(obs["tags"]).all()
    assert not np.isnan(obs["audits_emb"]).any()


def test_parsed_csv_runs_through_filter():
    cfg = Config()
    obs, T, _ = parse_observations_csv(observations_csv_template(cfg))
    assert T == cfg.geography.n_quarters  # template T matches cfg, so no resize needed here
    active = {"audits": True, "depots": True, "fires": True, "prices": True,
              "mailbox": True, "compactor": True, "tags": False}
    results = run_filter(obs, active, cfg, n_particles=500, seed=0)
    phi_med = results["quantiles"]["phi_bar"][:, 1]
    assert np.all((phi_med > 0) & (phi_med < 1))
    assert np.all(np.isfinite(results["n_eff"]))


def test_missing_columns_become_nan():
    csv_text = "quarter,fires,prices\n" + "\n".join(f"{t+1},{5+t%3},{100+t}" for t in range(6))
    obs, T, sales = parse_observations_csv(csv_text)
    assert T == 6
    assert sales is None
    assert np.isnan(obs["audits_emb"]).all()
    assert not np.isnan(obs["fires"]).any()


def test_blank_cells_become_nan():
    csv_text = "fires,prices\n4,\n,101\n6,102\n7,103\n"
    obs, T, _ = parse_observations_csv(csv_text)
    assert T == 4
    assert np.isnan(obs["prices"][0]) and np.isnan(obs["fires"][1])
    assert obs["fires"][0] == 4


@pytest.mark.parametrize("bad_csv,message_fragment", [
    ("fires\n3\n4\n5", "at least 4 quarters"),
    ("quarter,unknown_col\n1,2\n2,3\n3,4\n4,5", "No recognized channel columns"),
    ("fires,prices\n-1,100\n2,101\n3,102\n4,103", "cannot contain negative"),
    ("depots,prices\n0,100\n50,101\n60,102\n70,103", "must be positive"),
    ("sales,fires\n200,3\n,4\n210,5\n215,6", "fill it in every row"),
    ("sales,fires\n200,3\n-5,4\n210,5\n215,6", "'sales' must be positive"),
])
def test_malformed_uploads_raise_plain_errors(bad_csv, message_fragment):
    with pytest.raises(ValueError, match=message_fragment):
        parse_observations_csv(bad_csv)


def test_sales_override_drives_the_anchor():
    """A flat client sales series must pull the tonnage anchor to log(that level)."""
    from simulator import make_x_anchor_series

    cfg = Config()
    cfg.geography.n_quarters = 12
    cfg.geography.sales_override = tuple([500.0] * 12)
    anchors = make_x_anchor_series(cfg)
    # kappa sums to 1 over a flat history, so every anchor is exactly log(500)
    np.testing.assert_allclose(anchors, np.log(500.0), rtol=1e-10)

    cfg_default = Config()
    cfg_default.geography.n_quarters = 12
    assert not np.allclose(make_x_anchor_series(cfg_default), anchors)
