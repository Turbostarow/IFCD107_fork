from backend_utils import PickleRegressionModel


class TritonPythonModel(PickleRegressionModel):
    input_names = ("X",)
