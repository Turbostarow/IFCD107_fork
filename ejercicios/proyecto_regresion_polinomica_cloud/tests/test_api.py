import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from cloud_rf.app import create_app


def make_app(tmp_path):
    frame = pd.DataFrame({"superficie": [40, 50, 60, 70, 80, 90, 100, 110, 120, 130]})
    pipeline = make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestRegressor(n_estimators=5, random_state=4))
    pipeline.fit(frame, [10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    path = tmp_path / "model.joblib"
    joblib.dump({"pipeline": pipeline, "features": ["superficie"], "target": "y",
                 "metrics": {}, "n_rows": 10}, path)
    return create_app(str(path)).test_client()


def test_health_route(tmp_path):
    response = make_app(tmp_path).get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_predict_matches_vertex_request_format(tmp_path):
    response = make_app(tmp_path).post(
        "/predict", json={"instances": [{"superficie": 75}, {"superficie": 95}]}
    )
    assert response.status_code == 200
    assert len(response.json["predictions"]) == 2


def test_predict_rejects_wrong_feature_schema(tmp_path):
    response = make_app(tmp_path).post("/predict", json={"instances": [{"area": 75}]})
    assert response.status_code == 400
    assert "superficie" in response.json["missing"]
