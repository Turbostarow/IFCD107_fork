from backend_utils import PickleRegressionModel


class TritonPythonModel(PickleRegressionModel):
    input_names = (
        "distrito",
        "tipo_tramite",
        "superficie_solicitada_m2",
        "requerimientos_documentacion",
        "mes_presentacion",
        "x",
    )
    tabular = True
