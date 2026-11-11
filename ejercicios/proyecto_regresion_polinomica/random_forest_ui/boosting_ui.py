"""Controles principales y opciones completas en JSON de los boosters externos."""
import json
import streamlit as st

EXTERNOS = {'catboost', 'lightgbm', 'xgboost'}
REFERENCIAS = {
    'catboost': 'https://catboost.ai/docs/en/concepts/python-reference_catboostregressor',
    'lightgbm': 'https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html',
    'xgboost': 'https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor',
}
PERFILES = {
    'catboost': {
        'iterations': 100, 'depth': 6, 'learning_rate': .1, 'l2_leaf_reg': 3.,
        'loss_function': ['RMSE', 'MAE', 'Quantile:alpha=0.5', 'Huber:delta=1.0'],
        'random_strength': 1., 'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS', 'No'],
        'bagging_temperature': 1., 'subsample': .8, 'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
        'border_count': 254, 'min_data_in_leaf': 1, 'thread_count': 2, 'random_seed': 42,
    },
    'lightgbm': {
        'boosting_type': ['gbdt', 'dart', 'rf'], 'n_estimators': 100, 'num_leaves': 31,
        'max_depth': -1, 'learning_rate': .1, 'min_child_samples': 20, 'min_child_weight': .001,
        'min_split_gain': 0., 'subsample': .8, 'subsample_freq': 1, 'colsample_bytree': .9,
        'reg_alpha': 0., 'reg_lambda': 0., 'objective': ['regression', 'regression_l1', 'huber', 'quantile', 'poisson', 'tweedie'],
        'alpha': .9, 'random_state': 42, 'n_jobs': 2, 'importance_type': ['split', 'gain'],
    },
    'xgboost': {
        'n_estimators': 100, 'max_depth': 6, 'learning_rate': .1,
        'objective': ['reg:squarederror', 'reg:absoluteerror', 'reg:pseudohubererror', 'reg:quantileerror', 'count:poisson', 'reg:tweedie'],
        'booster': ['gbtree', 'dart', 'gblinear'], 'tree_method': ['hist', 'auto', 'approx', 'exact'],
        'min_child_weight': 1., 'gamma': 0., 'subsample': 1., 'colsample_bytree': 1.,
        'reg_alpha': 0., 'reg_lambda': 1., 'max_bin': 256, 'quantile_alpha': .5,
        'random_state': 42, 'n_jobs': 2,
    },
}


def crear_boosting(algoritmo, parametros, semilla=42):
    p = {k: (v[0] if isinstance(v, list) else v) for k, v in PERFILES[algoritmo].items()}
    p.update(parametros)
    if algoritmo == 'catboost':
        from catboost import CatBoostRegressor
        p.setdefault('verbose', False); p.setdefault('allow_writing_files', False); p.setdefault('task_type', 'CPU')
        if p.get('bootstrap_type') == 'Bayesian': p.pop('subsample', None)
        else: p.pop('bagging_temperature', None)
        if p.get('bootstrap_type') == 'No': p.pop('subsample', None)
        return CatBoostRegressor(**p)
    if algoritmo == 'lightgbm':
        from lightgbm import LGBMRegressor
        p.setdefault('verbosity', -1)
        if p.get('objective') not in {'huber', 'quantile'}: p.pop('alpha', None)
        return LGBMRegressor(**p)
    from xgboost import XGBRegressor
    p.setdefault('device', 'cpu')
    p.setdefault('verbosity', 0)
    if p.get('objective') != 'reg:quantileerror': p.pop('quantile_alpha', None)
    return XGBRegressor(**p)


def controles_boosting(algoritmo):
    from descripciones import EXPLICACIONES
    p = {}
    for nombre, valor in PERFILES[algoritmo].items():
        label, key = 'model__'+nombre, algoritmo+':'+nombre
        help_text = EXPLICACIONES.get(nombre, 'Parámetro del estimador; consulta Descripción y la referencia oficial.')
        if isinstance(valor, list):
            p[nombre] = st.selectbox(label, valor, key=key, help=help_text)
        else:
            minimo = -1 if nombre in {'max_depth', 'n_jobs', 'thread_count'} else 0
            p[nombre] = st.number_input(label, value=valor, min_value=int(minimo) if isinstance(valor, int) else float(minimo), key=key, help=help_text)
    texto = st.text_area('Parámetros adicionales · JSON', '{}', key=algoritmo+':extra',
                         help='Objeto con parámetros oficiales del constructor. Sobrescribe los controles con el mismo nombre. Las opciones incompatibles producen un error de entrenamiento.')
    try:
        extra = json.loads(texto)
        if not isinstance(extra, dict): raise ValueError('Introduce un objeto JSON.')
        p.update(extra)
    except ValueError as exc:
        st.error(str(exc)); st.stop()
    if algoritmo == 'catboost':
        if p.get('bootstrap_type') == 'Bayesian': p.pop('subsample', None)
        else: p.pop('bagging_temperature', None)
        if p.get('bootstrap_type') == 'No': p.pop('subsample', None)
    if algoritmo == 'xgboost' and p.get('objective') != 'reg:quantileerror':
        p.pop('quantile_alpha', None)
    st.caption('CPU. Parámetros adicionales: usa los nombres oficiales del constructor, sin model__. La prueba nunca se pasa como eval_set; la parada anticipada usa una validación interna del entrenamiento.')
    return p
