"""Entrena y valida el modelo que luego consume el contenedor de Vertex AI."""

import argparse
import json

from cloud_rf.model import train_from_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/random_forest_demo.csv")
    parser.add_argument("--output", default="artifacts/model.joblib")
    parser.add_argument("--target", default="y")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    artifact = train_from_csv(args.csv, args.output, args.target,
                              args.n_estimators, args.random_state)
    print(json.dumps({"rows": artifact["n_rows"], "features": artifact["features"],
                      "metrics_holdout": artifact["metrics"], "model": args.output}, indent=2))


if __name__ == "__main__":
    main()
