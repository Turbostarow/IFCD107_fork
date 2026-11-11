import unittest

import numpy as np
import pandas as pd
from curvas import calcular_curva, evaluador_tabular, evaluador_cnn, SUGERIDOS
from modelo import datos_ejemplo


class CurvasTests(unittest.TestCase):
    def test_folds_disjoint_and_learning_subsets_nested(self):
        llamadas = []
        def evaluar(train, val, p):
            self.assertFalse(set(train) & set(val))
            self.assertLess(max(np.concatenate([train, val])), 60)
            llamadas.append((set(train), set(val)))
            return 1., 2., len(train)
        tabla = calcular_curva(60, evaluar, {}, 'aprendizaje', [.3, .6, 1.], folds=2)
        self.assertTrue((tabla['Estado'] == 'ok').all())
        for fold in range(2):
            self.assertTrue(llamadas[fold][0] <= llamadas[2+fold][0] <= llamadas[4+fold][0])
            self.assertEqual(llamadas[fold][1], llamadas[4+fold][1])

    def test_chronological_folds_are_ordered(self):
        def evaluar(train, val, p):
            self.assertLess(max(train), min(val))
            return 1., 2., len(train)
        tabla = calcular_curva(60, evaluar, {}, 'aprendizaje', [.5, 1.], cronologico=True)
        self.assertTrue((tabla['Estado'] == 'ok').all())

    def test_failed_candidates_remain_visible(self):
        def evaluar(train, val, p):
            if p['alpha'] < 0: raise ValueError('alpha inválido')
            return 1., 2., len(train)
        tabla = calcular_curva(60, evaluar, {}, 'validacion', [-1., 1.], 'alpha')
        self.assertEqual((tabla['Estado'] == 'error').sum(), 2)
        self.assertTrue(tabla.loc[tabla['Estado'] == 'error', 'Aviso'].str.contains('alpha').all())

    def test_precomputed_model_is_refitted_for_each_fold(self):
        frame = datos_ejemplo().head(60)
        variables = list(frame.columns[:-1])
        evaluar = evaluador_tabular(frame, 'objetivo', variables, 'svr')
        tabla = calcular_curva(60, evaluar, {'kernel': 'precomputed'}, 'validacion', [.1, 1.], 'C')
        self.assertTrue((tabla['Estado'] == 'ok').all(), tabla.to_dict('records'))

    def test_no_external_test_data_can_influence_curve(self):
        frame = datos_ejemplo().head(80)
        variables = list(frame.columns[:-1])
        desarrollo = frame.iloc[:60].copy()
        params = {'random_state': 42}
        evaluar = evaluador_tabular(desarrollo, 'objetivo', variables, 'decision_tree')
        primero = calcular_curva(60, evaluar, params, 'validacion', [2, 5], 'max_depth')
        frame.iloc[60:] = 1000000
        segundo = calcular_curva(60, evaluador_tabular(frame.iloc[:60], 'objetivo', variables, 'decision_tree'), params, 'validacion', [2, 5], 'max_depth')
        pd.testing.assert_frame_equal(primero, segundo)


if __name__ == '__main__':
    unittest.main()
