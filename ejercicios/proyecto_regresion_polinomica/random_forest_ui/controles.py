"""Controles para todos los parámetros públicos de sklearn 1.8."""
import inspect
import json
import re

import joblib
import numpy as np
import streamlit as st
from sklearn.neighbors import VALID_METRICS
from sklearn.linear_model import LinearRegression, Ridge
from modelo import crear_estimador


def descripcion_parametros(estimador):
    doc = inspect.getdoc(type(estimador)) or ''
    section = doc.split('Parameters\n----------', 1)[-1].split('\nAttributes\n', 1)[0]
    bloques = re.split(r'\n(?=[a-zA-Z_]\w*\s*:)', '\n'+section.strip())
    result = {}
    for bloque in bloques:
        lineas = bloque.strip().splitlines()
        if lineas and ':' in lineas[0]:
            nombre, dominio = lineas[0].split(':', 1)
            result[nombre.strip()] = (dominio.strip(), '\n'.join(lineas[1:]).strip())
    return result


def cargar_objeto(nombre, key, funcion=False):
    archivo = st.file_uploader(f'{nombre} · .joblib de confianza', type=['joblib'], key=key)
    st.caption('Solo archivos de confianza: joblib.load puede ejecutar código. Los módulos de funciones o clases propias deben estar instalados también al usar el modelo exportado.')
    if archivo is None:
        st.info(f'Adjunta el archivo para {nombre}.')
        st.stop()
    try:
        objeto = joblib.load(archivo)
        if funcion and not callable(objeto):
            raise ValueError('El objeto debe ser una función invocable.')
        return objeto
    except Exception as exc:
        st.error(f'No se pudo cargar {nombre}: {exc}')
        st.stop()


def controles(algoritmo, variables, n_train, prefijo='model', estimador=None, anidado=False):
    if algoritmo in {'catboost', 'lightgbm', 'xgboost'} and not anidado:
        from boosting_ui import controles_boosting
        return controles_boosting(algoritmo)
    estimador = estimador if estimador is not None else crear_estimador(algoritmo)
    defaults = estimador.get_params(deep=False)
    docs = descripcion_parametros(estimador)
    p = {}
    # Los parámetros que condicionan otros se muestran primero.
    prioridad = ['bootstrap', 'kernel', 'algorithm', 'metric', 'loss', 'n_iter_no_change', 'estimator']
    nombres = [k for k in prioridad if k in defaults] + [k for k in defaults if k not in prioridad]
    for nombre in nombres:
        default = defaults[nombre]
        label = f'{prefijo}__{nombre}'
        key = f'{algoritmo}:{label}'
        help_text = '\n\n'.join(docs.get(nombre, ('', '')))

        def elegir(opciones, valor=default, sufijo='', disabled=False):
            index = next((i for i, v in enumerate(opciones) if type(v) is type(valor) and v == valor), 0)
            return st.selectbox(label+sufijo, opciones, index=index, key=key+sufijo, help=help_text, disabled=disabled,
                                format_func=lambda v: 'None · automático' if v is None else str(v))

        def numero(valor, minimo=None, entero=False, sufijo=''):
            valor = int(valor) if entero else float(valor)
            kwargs = {'value': valor, 'key': key+sufijo, 'help': help_text}
            if minimo is not None:
                kwargs['min_value'] = int(minimo) if entero else float(minimo)
            if not entero:
                kwargs.update(format='%.8g', step=0.000001 if 0 < abs(valor) < .001 else .01)
            return st.number_input(label+sufijo, **kwargs)

        def funcion(opciones, valor=default):
            opcion = elegir(opciones+['Función joblib'], valor)
            return cargar_objeto(label, key+':funcion', True) if opcion == 'Función joblib' else opcion

        if nombre in {'estimator', 'init'}:
            opciones = [None, 'Objeto joblib']
            if nombre == 'init':
                opciones += ['zero']
            opciones += ['LinearRegression', 'Ridge', 'Huber', 'DecisionTree']
            tipo = elegir(opciones)
            if tipo == 'Objeto joblib':
                p[nombre] = cargar_objeto(label, key+':objeto')
            elif tipo in {None, 'zero'}:
                p[nombre] = tipo
            else:
                base = {'LinearRegression': LinearRegression, 'Ridge': Ridge,
                        'Huber': lambda: crear_estimador('huber'),
                        'DecisionTree': lambda: crear_estimador('decision_tree')}[tipo]()
                with st.expander(f'Parámetros de {label}: {tipo}'):
                    base.set_params(**controles(algoritmo, variables, n_train, label, base, True))
                p[nombre] = base
        elif nombre in {'is_data_valid', 'is_model_valid'}:
            opcion = elegir([None, 'Función joblib'])
            p[nombre] = None if opcion is None else cargar_objeto(label, key+':funcion', True)
        elif nombre == 'kernel' and algoritmo == 'gaussian_process':
            from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel, ConstantKernel
            modo = elegir([None, 'RBF', 'Matern', 'RationalQuadratic', 'DotProduct', 'RBF + ruido', 'Objeto joblib'])
            if modo is None:
                p[nombre] = None
            elif modo == 'Objeto joblib':
                p[nombre] = cargar_objeto(label, key+':objeto')
            else:
                amplitud = numero(1., 1e-10, sufijo=' · amplitud')
                escala = numero(1., 1e-10, sufijo=' · length_scale')
                nu = st.selectbox(label+' · nu', [.5, 1.5, 2.5, float('inf')], index=1, key=key+':nu') if modo == 'Matern' else 1.5
                kernels = {'RBF': RBF(escala), 'Matern': Matern(escala, nu=nu),
                           'RationalQuadratic': RationalQuadratic(escala), 'DotProduct': DotProduct(),
                           'RBF + ruido': RBF(escala)+WhiteKernel()}
                p[nombre] = ConstantKernel(amplitud)*kernels[modo]
        elif nombre == 'optimizer' and algoritmo == 'gaussian_process':
            p[nombre] = funcion(['fmin_l_bfgs_b', None])
        elif nombre == 'hidden_layer_sizes':
            texto = st.text_input(label+' · tamaños separados por comas', ','.join(map(str, default)), key=key, help=help_text)
            try:
                capas = tuple(int(v.strip()) for v in texto.split(','))
                if not capas or min(capas) < 1: raise ValueError()
                p[nombre] = capas
            except ValueError:
                st.error('Cada capa necesita un número entero positivo, por ejemplo 64,32.'); st.stop()
        elif nombre == 'activation':
            p[nombre] = elegir(['identity', 'logistic', 'tanh', 'relu'])
        elif nombre == 'solver':
            p[nombre] = elegir(['lbfgs', 'sgd', 'adam'] if algoritmo == 'mlp' else ['highs', 'highs-ds', 'highs-ipm'])
        elif nombre == 'solver_options':
            texto = st.text_area(label+' · JSON o null', 'null', key=key, help=help_text)
            try:
                p[nombre] = json.loads(texto)
                if p[nombre] is not None and not isinstance(p[nombre], dict): raise ValueError('Se necesita un objeto o null.')
            except ValueError as exc:
                st.error(str(exc)); st.stop()
        elif nombre == 'learning_rate' and algoritmo == 'mlp':
            p[nombre] = elegir(['constant', 'invscaling', 'adaptive'])
        elif nombre == 'batch_size':
            modo = elegir(['auto', 'Número'])
            p[nombre] = 'auto' if modo == 'auto' else numero(32, 1, True, ' · número')
        elif nombre == 'kernel':
            p[nombre] = funcion(['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
        elif nombre == 'weights':
            p[nombre] = funcion(['uniform', 'distance'])
        elif nombre == 'metric':
            motor = p.get('algorithm', 'auto')
            metricas = sorted(set(VALID_METRICS['brute' if motor == 'auto' else motor]))
            if len(variables) != 2:
                metricas = [m for m in metricas if m != 'haversine']
            p[nombre] = funcion(metricas)
        elif nombre == 'metric_params':
            texto = st.text_area(label+' · objeto JSON o null', value='null', key=key, help=help_text)
            try:
                valor = json.loads(texto)
                if valor is not None and not isinstance(valor, dict):
                    raise ValueError('Se necesita un objeto JSON o null.')
                p[nombre] = valor
            except (ValueError, TypeError) as exc:
                st.error(f'{label}: {exc}')
                st.stop()
        elif nombre == 'monotonic_cst':
            activo = st.checkbox(label+' · activar', key=key, help=help_text)
            p[nombre] = [st.selectbox(f'{label} · {v}', [0, 1, -1], key=key+':'+v) for v in variables] if activo else None
        elif nombre == 'loss':
            if algoritmo == 'mlp':
                p[nombre] = elegir(['squared_error', 'poisson'])
            elif algoritmo == 'ransac':
                p[nombre] = funcion(['absolute_error', 'squared_error'])
            else:
                p[nombre] = elegir(['squared_error', 'absolute_error', 'huber', 'quantile'])
        elif nombre == 'criterion':
            opciones = ['friedman_mse', 'squared_error'] if algoritmo == 'gradient_boosting' and not anidado else ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
            p[nombre] = elegir(opciones)
        elif nombre == 'splitter':
            p[nombre] = elegir(['best', 'random'])
        elif nombre == 'algorithm':
            p[nombre] = elegir(['auto', 'ball_tree', 'kd_tree', 'brute'])
        elif nombre == 'gamma':
            modo = elegir(['scale', 'auto', 'Valor positivo'], default, disabled=p.get('kernel') in {'linear', 'precomputed'})
            p[nombre] = numero(.01, 1e-12, sufijo=' · valor') if modo == 'Valor positivo' else modo
        elif nombre == 'max_features':
            modo = elegir([None, 'sqrt', 'log2', 'Número', 'Fracción'], 'Fracción' if isinstance(default, float) else default)
            if modo == 'Número':
                p[nombre] = numero(1, 1, True, ' · número')
            elif modo == 'Fracción':
                p[nombre] = st.slider(label+' · fracción', .01, 1., default if isinstance(default, float) else 1., .01, key=key+':fraccion')
            else:
                p[nombre] = modo
        elif nombre in {'min_samples_split', 'min_samples_leaf', 'min_samples', 'max_samples'}:
            opciones = ['Número', 'Fracción']
            if nombre in {'min_samples', 'max_samples'}:
                opciones.insert(0, None)
            modo = elegir(opciones, None if default is None else 'Fracción' if isinstance(default, float) else 'Número', sufijo=' · unidad', disabled=nombre == 'max_samples' and not p.get('bootstrap', True))
            if modo is None or (nombre == 'max_samples' and not p.get('bootstrap', True)):
                p[nombre] = None
            elif modo == 'Fracción':
                p[nombre] = st.slider(label+' · fracción', .01, 1., default if isinstance(default, float) else .5, .01, key=key+':fraccion')
            else:
                p[nombre] = numero(default if isinstance(default, int) else max(2, len(variables)+1), 2 if nombre == 'min_samples_split' else 1, True, ' · número')
        elif nombre in {'max_skips', 'stop_n_inliers', 'stop_score'}:
            ilimitado = st.checkbox(label+' · infinito', True, key=key+':inf', help=help_text)
            p[nombre] = float('inf') if ilimitado else numero(0, None if nombre == 'stop_score' else 0, nombre != 'stop_score')
        elif nombre in {'max_depth', 'max_leaf_nodes', 'n_iter_no_change', 'n_subsamples', 'alpha_init', 'lambda_init', 'residual_threshold', 'n_jobs', 'random_state', 'n_targets'}:
            automatico = st.checkbox(label+' · None', default is None, key=key+':none', help=help_text)
            if automatico:
                p[nombre] = None
            else:
                entero = nombre not in {'alpha_init', 'lambda_init', 'residual_threshold'}
                minimo = -1 if nombre == 'n_jobs' else 0 if nombre in {'random_state', 'residual_threshold'} else 2 if nombre == 'max_leaf_nodes' else 1e-12 if not entero else 1
                p[nombre] = numero(default if default is not None else 1 if nombre != 'n_subsamples' else len(variables)+1, minimo, entero)
                if nombre == 'n_jobs' and p[nombre] == 0:
                    st.error('n_jobs no puede ser cero.'); st.stop()
        elif nombre == 'oob_score':
            modo = elegir([False, True, 'Función joblib'], disabled=not p.get('bootstrap', True))
            p[nombre] = False if not p.get('bootstrap', True) else cargar_objeto(label, key+':funcion', True) if modo == 'Función joblib' else modo
        elif isinstance(default, bool):
            disabled = nombre == 'oob_score' and not p.get('bootstrap', True)
            p[nombre] = st.checkbox(label, default, key=key, help=help_text, disabled=disabled) and not disabled
        elif nombre in {'subsample', 'validation_fraction', 'stop_probability', 'min_weight_fraction_leaf', 'quantile', 'beta_1', 'beta_2', 'momentum'} or (nombre == 'alpha' and algoritmo == 'gradient_boosting' and not anidado):
            minimo = 0. if nombre in {'stop_probability', 'min_weight_fraction_leaf'} else .001
            maximo = .5 if nombre == 'min_weight_fraction_leaf' else .999 if nombre in {'alpha', 'validation_fraction', 'quantile', 'beta_1', 'beta_2'} else 1.
            p[nombre] = st.slider(label, minimo, maximo, float(default), .001, key=key, help=help_text)
        elif isinstance(default, (int, float)):
            entero = isinstance(default, int)
            minimo = -1 if nombre == 'max_iter' and algoritmo == 'svr' else 0 if nombre in {'alpha', 'alpha_1', 'alpha_2', 'lambda_1', 'lambda_2', 'epsilon', 'min_impurity_decrease', 'ccp_alpha', 'degree', 'verbose', 'tol', 'n_restarts_optimizer'} else None if nombre == 'coef0' else 1 if entero else 1e-12
            if nombre == 'epsilon' and algoritmo == 'huber':
                minimo = 1.
            p[nombre] = numero(default, minimo, entero)
        else:
            raise ValueError(f'Falta un control para {label}.')
    return p
