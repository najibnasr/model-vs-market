import numpy as np

from mvm.sports.foot import build, market_baseline


def test_market_baseline_runs_on_fixture_data(cfg):
    build.write_outputs(build.build(cfg), cfg)
    cfg["evaluation"] = {"eval_first": "2012-2013"}
    df = market_baseline.run(cfg)
    assert set(df.season) == {"2020-2021"}  # 2005-06 exclu : burn-in, pas de Pinnacle
    assert df["log_loss_pin_close"].notna().all()
    assert np.isfinite(df["rps_pin_close"]).all()
    summary = market_baseline.summarize(df)
    assert summary["n"].sum() == len(df)
