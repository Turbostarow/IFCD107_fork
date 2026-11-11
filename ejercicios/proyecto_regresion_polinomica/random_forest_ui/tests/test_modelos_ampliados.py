import io
import unittest
import warnings

import joblib
import numpy as np
from modelo import datos_ejemplo, entrenar


class ModelosAmpliadosTests(unittest.TestCase):
    def test_predictions_roundtrip_and_test_alignment(self):
        frame = datos_ejemplo().head(80)
        variables = list(frame.columns[:-1])
        candidatos = [('catboost', {'iterations': 10}), ('gaussian_process', {}),
                      ('gradient_boosting', {'n_estimators': 10}), ('lightgbm', {'n_estimators': 10}),
                      ('mlp', {'hidden_layer_sizes': (8,), 'max_iter': 50}), ('quantile', {}),
                      ('xgboost', {'n_estimators': 10})]
        for algoritmo, parametros in candidatos:
            with self.subTest(algoritmo=algoritmo), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                modelo, resultados, test, _ = entrenar(frame, 'objetivo', variables, parametros, algoritmo=algoritmo)
                self.assertEqual(len(test), 5)
                np.testing.assert_allclose(resultados['observado'], frame.loc[resultados.index, 'objetivo'])
                self.assertFalse(set(modelo.training_indices_) & set(resultados.index))
                buffer = io.BytesIO(); joblib.dump(modelo, buffer, compress=3); buffer.seek(0)
                np.testing.assert_allclose(joblib.load(buffer).predict(frame.loc[resultados.index, variables]), resultados['predicho'], rtol=1e-5)

    def test_early_stopping_does_not_use_test(self):
        frame = datos_ejemplo().head(80)
        variables = list(frame.columns[:-1])
        for algoritmo, parametros in [('catboost', {'iterations': 20, 'early_stopping_rounds': 3}),
                                     ('lightgbm', {'n_estimators': 20, 'early_stopping_round': 3}),
                                     ('xgboost', {'n_estimators': 20, 'early_stopping_rounds': 3})]:
            with self.subTest(algoritmo=algoritmo):
                modelo, resultados, _, _ = entrenar(frame, 'objetivo', variables, parametros, algoritmo=algoritmo)
                train, val, test = map(set, [modelo.training_indices_, modelo.validation_indices_, resultados.index])
                self.assertFalse(train & val or train & test or val & test)
                self.assertEqual(len(train | val | test), len(frame))


if __name__ == '__main__':
    unittest.main()
