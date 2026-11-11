"""Entrenamiento y métricas independientes de la interfaz."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, RANSACRegressor, HuberRegressor, TheilSenRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
from functools import partial


def datos_ejemplo():
    rng = np.random.default_rng(42)
    x = rng.uniform(-3, 3, (600, 4))
    df = pd.DataFrame(x, columns=['temperatura', 'humedad', 'altura', 'radiacion'])
    df['objetivo'] = 20 + 4*x[:, 0] + 3*x[:, 1]**2 + 5*np.sin(x[:, 2]) + rng.normal(0, 2, len(x))
    return df


def metricas(real, pred):
    real, pred = np.asarray(real), np.asarray(pred)
    pearson = float(np.corrcoef(real, pred)[0, 1]) if np.std(real) > 0 and np.std(pred) > 0 else float('nan')
    return {'R²': r2_score(real, pred), 'RMSE': np.sqrt(mean_squared_error(real, pred)),
            'MAE': mean_absolute_error(real, pred), 'Pearson': pearson,
            'Bias': float(np.mean(pred-real))}


def crear_estimador(algoritmo, parametros=None, semilla=42):
    clases = {
        'bayesian_ridge': BayesianRidge, 'random_forest': RandomForestRegressor,
        'ransac': RANSACRegressor, 'huber': HuberRegressor,
        'theil_sen': TheilSenRegressor, 'decision_tree': DecisionTreeRegressor,
        'gradient_boosting': GradientBoostingRegressor, 'svr': SVR, 'knn': KNeighborsRegressor,
        'mlp': MLPRegressor, 'gaussian_process': GaussianProcessRegressor, 'quantile': QuantileRegressor,
    }
    if algoritmo in {'catboost', 'lightgbm', 'xgboost'}:
        from boosting_ui import crear_boosting
        return crear_boosting(algoritmo, parametros or {}, semilla)
    if algoritmo not in clases:
        raise ValueError('Algoritmo desconocido.')
    parametros = dict(parametros or {})
    if algoritmo in {'ransac', 'theil_sen', 'decision_tree', 'gradient_boosting', 'mlp', 'gaussian_process'}:
        parametros.setdefault('random_state', semilla)
    if algoritmo == 'huber':
        parametros.setdefault('max_iter', 1000)
    if algoritmo == 'theil_sen':
        parametros.setdefault('max_subpopulation', 500)
    return clases[algoritmo](**parametros)


def entrenar(df, objetivo, variables, parametros, prueba=0.25, semilla=42, algoritmo='random_forest', particion=None):
    datos = df[variables + [objetivo]].replace([np.inf, -np.inf], np.nan).dropna(subset=[objetivo])
    if len(datos) < 10:
        raise ValueError('Se necesitan al menos 10 filas con objetivo válido.')
    if particion is None:
        x_train, x_test, y_train, y_test = train_test_split(
            datos[variables], datos[objetivo], test_size=prueba, random_state=semilla)
    else:
        fit, validacion = (np.asarray(i, dtype=int) for i in particion)
        if not len(fit) or len(validacion) < 2 or np.intersect1d(fit, validacion).size:
            raise ValueError('Las particiones deben ser no vacías, disjuntas y tener al menos dos filas de evaluación.')
        x_train, x_test = datos[variables].iloc[fit], datos[variables].iloc[validacion]
        y_train, y_test = datos[objetivo].iloc[fit], datos[objetivo].iloc[validacion]
    if x_train.isna().all().any():
        raise ValueError('Hay variables sin datos en entrenamiento. Deselecciónalas.')
    if parametros.get('criterion') == 'poisson' and (y_train.min() < 0 or y_train.sum() <= 0):
        raise ValueError('Poisson necesita objetivos no negativos con suma positiva en entrenamiento.')
    if (parametros.get('loss') == 'poisson' or parametros.get('objective') in {'poisson', 'count:poisson', 'reg:tweedie', 'tweedie'}) and (y_train.min() < 0 or y_train.sum() <= 0):
        raise ValueError('La pérdida elegida requiere objetivos no negativos con suma positiva.')
    estimador = crear_estimador(algoritmo, parametros, semilla)
    if algoritmo == 'knn' and estimador.n_neighbors > len(x_train):
        raise ValueError(f'n_neighbors no puede superar las {len(x_train)} filas de entrenamiento.')
    pasos = [SimpleImputer(strategy='median')]
    if algoritmo not in {'random_forest', 'bayesian_ridge'}:
        pasos.append(StandardScaler())
    precomputado = (algoritmo == 'svr' and parametros.get('kernel') == 'precomputed') or (algoritmo == 'knn' and parametros.get('metric') == 'precomputed')
    if precomputado:
        if algoritmo == 'knn' and parametros.get('algorithm', 'auto') not in {'auto', 'brute'}:
            raise ValueError('La métrica precomputed requiere algorithm=auto o brute.')
        # Matriz train/train y, al predecir, matriz nuevas/train. Se conserva la
        # referencia de entrenamiento en un partial serializable con joblib.
        referencia = make_pipeline(*pasos).fit_transform(x_train)
        funcion = partial(pairwise_kernels, Y=referencia, metric='linear') if algoritmo == 'svr' else partial(pairwise_distances, Y=referencia, metric='euclidean')
        pasos.append(FunctionTransformer(funcion))
    modelo = make_pipeline(*pasos, estimador)
    parada_externa = algoritmo in {'catboost', 'lightgbm', 'xgboost'} and any(parametros.get(k) for k in ['early_stopping_rounds', 'early_stopping_round', 'od_wait', 'od_type', 'use_best_model'])
    if parada_externa:
        x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=.2, random_state=semilla)
        preproceso = make_pipeline(*pasos)
        preproceso.fit(x_fit)
        kwargs = {f'{modelo.steps[-1][0]}__eval_set': [(preproceso.transform(x_val), y_val)]}
        if algoritmo == 'xgboost':
            kwargs[f'{modelo.steps[-1][0]}__verbose'] = False
        modelo.fit(x_fit, y_fit, **kwargs)
        modelo.validation_indices_ = x_val.index.tolist()
        x_train, y_train = x_fit, y_fit
    else:
        modelo.fit(x_train, y_train)
    modelo.training_indices_ = x_train.index.tolist()
    pred = modelo.predict(x_test)
    resultados = pd.DataFrame({'observado': y_test, 'predicho': pred,
                               'residuo': y_test-pred})
    return modelo, resultados, metricas(y_test, pred), metricas(y_train, modelo.predict(x_train))
