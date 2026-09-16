from quant.backtest import run_backtest, score_panel
from quant.config import load_config
from quant.demo import synthetic_prices
from quant.features import build_features


def test_demo_pipeline():
    cfg=load_config("configs/base.yaml")
    prices,b=synthetic_prices(n_stocks=20,n_days=300)
    panel=build_features(prices,cfg["features"]["enabled"])
    scored=score_panel(panel,cfg)
    eq,_tr,metrics=run_backtest(scored,b,cfg)
    assert len(eq)>100
    assert "max_drawdown" in metrics
