import unittest
import numpy as np
from geometria import datos_prueba, corte_modelo


class GeometriaTests(unittest.TestCase):
    def test_test_points_preserve_row_order_and_saved_predictions(self):
        x = np.arange(30., dtype=float).reshape(10, 3)
        y = np.arange(10.)*2
        indices = [8, 1, 6]
        pred = [15., 3., 11.]
        tabla = datos_prueba(x, y, indices, pred, ['a', 'b', 'c'])
        np.testing.assert_array_equal(tabla[['a','b','c']], x[indices])
        np.testing.assert_array_equal(tabla['Observado'], y[indices])
        np.testing.assert_array_equal(tabla['Predicho'], pred)
        np.testing.assert_array_equal(tabla['Residuo'], y[indices]-pred)

    def test_linear_plane_uses_real_features_and_training_medians(self):
        x = np.random.default_rng(42).normal(size=(60,3))
        train = np.arange(40)
        predict = lambda a: 2*a[:,0]-3*a[:,1]+5*a[:,2]+7
        gx, gy, z, base = corte_modelo(x, train, 0, 1, predict)
        np.testing.assert_allclose(z, 2*gx-3*gy+5*np.median(x[train,2])+7)
        x[40:] += 1000
        _, _, z2, base2 = corte_modelo(x, train, 0, 1, predict)
        np.testing.assert_array_equal(z, z2)
        np.testing.assert_array_equal(base, base2)

    def test_nonlinear_surface_is_model_prediction(self):
        x = np.array([[0.,1.],[1.,2.],[2.,3.],[3.,4.]])
        predict = lambda a: a[:,0]**2+np.sin(a[:,1])
        gx, gy, z, _ = corte_modelo(x, [0,1,2], 0, 1, predict, 15)
        np.testing.assert_allclose(z, gx**2+np.sin(gy))

    def test_one_feature_curve(self):
        x = np.arange(10.).reshape(-1,1)
        gx, gy, z, _ = corte_modelo(x, np.arange(7), 0, None, lambda a: 3*a[:,0]+1, 20)
        self.assertIsNone(gy)
        np.testing.assert_allclose(z, 3*gx+1)


if __name__ == '__main__':
    unittest.main()
