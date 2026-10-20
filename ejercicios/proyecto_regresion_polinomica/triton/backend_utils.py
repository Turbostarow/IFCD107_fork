"""Common Triton Python-backend runner for the project's sklearn artifacts."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import triton_python_backend_utils as pb_utils


class PickleRegressionModel:
    """Loads one trusted, pre-trained estimator and serves batched requests."""

    input_names: tuple[str, ...] = ("X",)
    tabular = False

    def initialize(self, args: dict) -> None:
        model_dir = Path(args["model_repository"]) / str(args["model_version"])
        with (model_dir / "model.pkl").open("rb") as stream:
            self.model = pickle.load(stream)
        model_config = json.loads(args["model_config"])
        self.output_name = model_config["output"][0]["name"]

    def _inputs_to_features(self, request):
        columns = {}
        for name in self.input_names:
            values = pb_utils.get_input_tensor_by_name(request, name).as_numpy().reshape(-1)
            if name in {"distrito", "tipo_tramite"}:
                values = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
            else:
                values = np.asarray(values, dtype=np.float32)
            columns[name] = values
        if self.tabular:
            lengths = {len(values) for values in columns.values()}
            if len(lengths) != 1:
                raise ValueError("Todas las entradas deben contener el mismo número de filas.")
            return pd.DataFrame(columns, columns=list(self.input_names))
        values = columns[self.input_names[0]]
        if not np.isfinite(values).all():
            raise ValueError("La entrada contiene valores no finitos.")
        return np.asarray(values, dtype=np.float32).reshape(-1, 1)

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                features = self._inputs_to_features(request)
                prediction = np.asarray(self.model.predict(features), dtype=np.float32).reshape(-1, 1)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor(self.output_name, prediction)]
                    )
                )
            except Exception as error:
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error)))
                )
        return responses
