from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .models.factory import create
from .utils import rank_ic


def time_split(df: pd.DataFrame, train_ratio: float, valid_ratio: float, purge_days: int = 0):
    dates = np.array(sorted(pd.to_datetime(df["date"].unique())))
    n = len(dates)
    a = max(2, int(n * train_ratio))
    b = max(a + 2, int(n * (train_ratio + valid_ratio)))
    a = min(a, n - 4); b = min(b, n - 2)
    train_end = max(1, a - purge_days)
    valid_end = max(a + 1, b - purge_days)
    train_dates = set(dates[:train_end])
    valid_dates = set(dates[a:valid_end])
    test_dates = set(dates[b:])
    return df[df.date.isin(train_dates)], df[df.date.isin(valid_dates)], df[df.date.isin(test_dates)]


def _version_id(cfg: dict) -> str:
    raw = json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str).encode()
    h = hashlib.sha256(raw).hexdigest()[:8]
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{h}"  # noqa: DTZ005 - preserve local artifact IDs


def train_model(dataset: pd.DataFrame, cfg: dict, artifact_dir: str = "artifacts") -> dict:
    feature_names = cfg["features"]["enabled"]
    clean = dataset.dropna(subset=feature_names + ["label"]).copy()
    train, valid, test = time_split(clean, cfg["training"]["train_ratio"], cfg["training"]["valid_ratio"], purge_days=int(cfg["label"]["horizon"]))
    if min(len(train), len(valid), len(test)) == 0:
        raise RuntimeError("dataset too small for train/valid/test split")

    model_name = cfg["model"]["name"]
    model = create(model_name, cfg["model"].get("params", {}))
    model.fit(train[feature_names], train["label"])
    valid = valid.copy(); test = test.copy()
    valid["prediction"] = model.predict(valid[feature_names])
    test["prediction"] = model.predict(test[feature_names])

    model_id = _version_id(cfg)
    metrics = {
        "valid_rank_ic": rank_ic(valid, "prediction", "label"),
        "test_rank_ic": rank_ic(test, "prediction", "label"),
        "train_rows": len(train), "valid_rows": len(valid), "test_rows": len(test),
    }

    root = Path(artifact_dir)
    version_dir = root / model_name / model_id
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / "model.joblib"
    model.save(model_path)
    (version_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    latest_path = root / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8")) if latest_path.exists() else {}
    latest[model_name] = str(model_path.as_posix())
    latest_path.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import mlflow
        mlflow.set_tracking_uri(cfg["training"]["mlflow_uri"])
        mlflow.set_experiment(cfg["training"]["experiment_name"])
        with mlflow.start_run(run_name=f"{model_name}-{model_id}"):
            mlflow.log_params({
                "model": model_name,
                "model_id": model_id,
                "features": len(feature_names),
                "label_horizon": cfg["label"]["horizon"],
            })
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if "rows" not in k})
            mlflow.log_artifacts(str(version_dir))
    except Exception as e:  # noqa: BLE001 - optional MLflow telemetry must not block model training
        print(f"[WARN] MLflow logging skipped: {e}")

    return {**metrics, "model_id": model_id, "model_path": str(model_path)}
