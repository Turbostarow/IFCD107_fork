import pandas as pd
import pytest

from cloud_rf.model import build_pipeline, train_from_csv


def test_pipeline_trains_with_numeric_and_categorical_features():
    frame = pd.DataFrame({"size": [40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
                          "zone": ["A", "B"] * 5})
    target = pd.Series([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    model = build_pipeline(frame, n_estimators=10)
    model.fit(frame, target)
    assert len(model.predict(frame)) == len(frame)


def test_training_writes_artifact_and_metrics(tmp_path):
    csv_path = tmp_path / "training.csv"
    frame = pd.DataFrame({"x": range(30), "group": ["a", "b", "c"] * 10,
                          "y": [2 * x for x in range(30)]})
    frame.to_csv(csv_path, index=False)
    output = tmp_path / "artifact" / "model.joblib"
    result = train_from_csv(csv_path, output, n_estimators=10)
    assert output.is_file()
    assert set(result["metrics"]) == {"r2", "rmse", "mae"}
    assert result["features"] == ["x", "group"]


def test_training_rejects_missing_target(tmp_path):
    csv_path = tmp_path / "training.csv"
    pd.DataFrame({"x": range(12)}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="objetivo"):
        train_from_csv(csv_path, tmp_path / "model.joblib")
