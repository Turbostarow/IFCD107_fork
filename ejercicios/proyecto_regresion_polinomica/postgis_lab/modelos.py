import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

FEATURES = ['superficie', 'habitaciones', 'antiguedad', 'distancia_centro_km']


def entrenar(df, algoritmo='Lineal', alpha=1.0, l1_ratio=0.5, test_size=0.25):
    if len(df) < 40:
        raise ValueError('Se necesitan al menos 40 viviendas para el ejercicio.')
    modelos = {'Lineal': LinearRegression(), 'Ridge': Ridge(alpha=alpha),
               'Lasso': Lasso(alpha=alpha, max_iter=20000),
               'Elastic Net': ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000)}
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    pipe = make_pipeline(StandardScaler(), modelos[algoritmo])
    pipe.fit(train[FEATURES], train.precio)
    pred = pipe.predict(test[FEATURES])
    error = pred - test.precio.to_numpy()
    pearson = float(np.corrcoef(test.precio, pred)[0, 1]) if np.std(pred) > 0 and test.precio.std() > 0 else None
    metrics = {'R²': r2_score(test.precio, pred),
               'RMSE': float(np.sqrt(mean_squared_error(test.precio, pred))),
               'MAE': mean_absolute_error(test.precio, pred),
               'Pearson': pearson, 'Bias (predicho − real)': float(error.mean())}
    results = test.copy()
    results['predicho'] = pred
    results['residuo'] = test.precio - pred
    scaler, model = pipe.steps[0][1], pipe.steps[1][1]
    coef = model.coef_ / scaler.scale_
    coefficients = pd.DataFrame({'variable': FEATURES, 'coeficiente_unidades_originales': coef})
    intercept = float(model.intercept_ - np.dot(coef, scaler.mean_))
    return pipe, metrics, results, coefficients, intercept
