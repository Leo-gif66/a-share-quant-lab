from pathlib import Path
import yaml


class Universe:

    def __init__(self, config_path="configs/universe.yaml"):
        self.config_path = Path(config_path)
        self.data = self._load()

    def _load(self):
        with open(
            self.config_path,
            encoding="utf-8"
        ) as f:
            return yaml.safe_load(f)

    def stocks(self):
        result = []

        for category, items in self.data["stocks"].items():

            for stock in items:
                result.append({
                    "code": stock["code"],
                    "name": stock["name"],
                    "category": category
                })

        return result


if __name__ == "__main__":

    universe = Universe()

    for stock in universe.stocks():
        print(
            stock["code"],
            stock["name"],
            stock["category"]
        )