"""API HTTP compatible con el protocolo predict de Vertex AI."""

from __future__ import annotations

import os

import pandas as pd
from flask import Flask, jsonify, request

from cloud_rf.model import load_artifact


def create_app(model_path: str | None = None) -> Flask:
    """Crea la API; Vertex AI envía las predicciones a la ruta /predict."""
    app = Flask(__name__)
    artifact_path = model_path or os.environ.get("MODEL_PATH", "/app/artifacts/model.joblib")
    artifact = load_artifact(artifact_path)

    @app.get("/health")
    def health():
        return jsonify(status="ok", algorithm="RandomForestRegressor")

    @app.post("/predict")
    def predict():
        body = request.get_json(silent=True)
        instances = body.get("instances") if isinstance(body, dict) else None
        if not isinstance(instances, list) or not instances:
            return jsonify(error="JSON debe incluir una lista no vacía 'instances'."), 400
        if any(not isinstance(row, dict) for row in instances):
            return jsonify(error="Cada elemento de instances debe ser un objeto JSON."), 400

        expected = artifact["features"]
        for row in instances:
            missing = [name for name in expected if name not in row]
            extra = [name for name in row if name not in expected]
            if missing or extra:
                return jsonify(error="Las columnas no coinciden con el modelo.",
                               missing=missing, extra=extra), 400
        try:
            values = pd.DataFrame(instances, columns=expected)
            predictions = artifact["pipeline"].predict(values).astype(float).tolist()
        except (TypeError, ValueError) as exc:
            return jsonify(error=f"No se pudieron interpretar las instancias: {exc}"), 400
        return jsonify(predictions=predictions)

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
