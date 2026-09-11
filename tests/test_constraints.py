import pytest

from quant.portfolio.industry_neutral import PortfolioConstraints


def test_portfolio_constraints_load_from_yaml_and_validate(tmp_path):
    config = tmp_path / "constraints.yaml"
    config.write_text(
        "max_industry_weight: 0.25\nmin_industry_count: 5\nmax_stock_weight: 0.10\n",
        encoding="utf-8",
    )

    constraints = PortfolioConstraints.from_yaml(config)

    assert constraints.max_industry_weight == 0.25
    assert constraints.min_industry_count == 5
    assert constraints.max_stock_weight == 0.10
    with pytest.raises(ValueError, match="max_stock_weight"):
        PortfolioConstraints(max_stock_weight=0)


def test_portfolio_constraints_reject_unknown_configuration(tmp_path):
    config = tmp_path / "constraints.yaml"
    config.write_text("max_industry_weight: 0.25\nunsupported: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown portfolio constraint"):
        PortfolioConstraints.from_yaml(config)

