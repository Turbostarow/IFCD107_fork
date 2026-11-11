"""Ejecutar con Python en la imagen del laboratorio (PyTorch CPU)."""
import io
import unittest

import numpy as np
import torch
from cnn_modelo import ejemplo_1d, ejemplo_2d, zip_ejemplo, leer_imagenes, entrenar_cnn, predecir_cnn, crear_red


CONFIG = dict(layers=2, filters=8, kernel_size=3, stride=1, dilation=1, activation='relu',
              pooling='max', dropout=0., hidden_units=16, optimizer='Adam', learning_rate=.003,
              weight_decay=0., momentum=.9, loss='MSE', epochs=3, batch_size=16, patience=2, min_delta=.0001)


class CNNTests(unittest.TestCase):
    def test_roundtrip_weights_and_predictions_both_dimensions(self):
        frame = ejemplo_1d().head(80)
        x1 = frame.iloc[:, :-1].to_numpy(dtype='float32')[:, None, :]
        x2, y2, _ = ejemplo_2d()
        for dimension, x, y in [(1, x1, frame['objetivo'].to_numpy()), (2, x2[:60], y2[:60])]:
            with self.subTest(dimension=dimension):
                paquete, resultados, test, _, historia = entrenar_cnn(x, y, dimension, CONFIG)
                buffer = io.BytesIO(); torch.save(paquete, buffer); buffer.seek(0)
                restored = torch.load(buffer, weights_only=True, map_location='cpu')
                pred = predecir_cnn(restored, x[resultados['indice'].to_numpy()])
                np.testing.assert_allclose(pred, resultados['predicho'], rtol=1e-5, atol=1e-5)
                self.assertEqual(len(test), 5)
                self.assertTrue(np.isfinite(resultados[['observado', 'predicho', 'residuo']]).all().all())
                grupos = [set(restored['indices'][key]) for key in ['train', 'validation', 'test']]
                self.assertFalse(grupos[0] & grupos[1] or grupos[0] & grupos[2] or grupos[1] & grupos[2])
                self.assertGreater(len(historia), 0)

    def test_test_data_do_not_affect_training(self):
        frame = ejemplo_1d().head(60)
        x = frame.iloc[:, :-1].to_numpy(dtype='float32', copy=True)[:, None, :]
        y = frame['objetivo'].to_numpy(dtype='float32', copy=True)
        primero, *_ = entrenar_cnn(x, y, 1, CONFIG)
        test = primero['indices']['test']
        x[test] += 100
        y[test] += 100
        segundo, *_ = entrenar_cnn(x, y, 1, CONFIG)
        np.testing.assert_array_equal(primero['mean'], segundo['mean'])
        self.assertEqual(primero['target_mean'], segundo['target_mean'])
        for nombre in primero['state_dict']:
            torch.testing.assert_close(primero['state_dict'][nombre], segundo['state_dict'][nombre])

    def test_zip_gray_and_rgb(self):
        for color, canales in [('L', 1), ('RGB', 3)]:
            x, y, nombres = leer_imagenes(zip_ejemplo(), 16, color)
            self.assertEqual(x.shape, (160, canales, 16, 16))
            self.assertEqual(len(y), len(nombres))
            self.assertGreaterEqual(float(x.min()), 0.)
            self.assertLessEqual(float(x.max()), 1.)

    def test_small_inputs_pooling_and_stride(self):
        for dimension in [1, 2]:
            for pooling in ['max', 'avg', 'none']:
                config = dict(CONFIG, layers=4, stride=3, dilation=3, pooling=pooling)
                red = crear_red(dimension, 1, config)
                self.assertEqual(tuple(red(torch.zeros((2, 1)+((1,)*dimension))).shape), (2, 1))


if __name__ == '__main__':
    unittest.main()
