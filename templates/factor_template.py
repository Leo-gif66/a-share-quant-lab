"""Copy this file into src/quant/factors/ and rename the class/name."""
from quant.factors.base import Factor
from quant.factors.registry import register


class MyCustomFactor(Factor):
    name = "my_custom_factor"

    def calculate(self, df):
        # Example only: replace with your own point-in-time factor logic.
        return df["close"].pct_change(15)


register(MyCustomFactor())
