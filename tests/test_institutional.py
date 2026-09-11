from quant.portfolio import InstitutionalPortfolioBacktestEngine
from quant.risk import AdvancedRiskController


class _Universe:
    def stocks(self):
        return [{"code": "000001", "name": "Test", "market": "SZ", "sector": "Test"}]


def test_institutional_engine_reuses_industry_engine_with_advanced_risk_dependency(tmp_path):
    engine = InstitutionalPortfolioBacktestEngine(
        features_dir=tmp_path / "features", raw_dir=tmp_path / "raw", universe=_Universe()
    )

    assert isinstance(engine.risk, AdvancedRiskController)
    assert engine.risk.settings.max_position == engine.constraints.max_stock_weight
