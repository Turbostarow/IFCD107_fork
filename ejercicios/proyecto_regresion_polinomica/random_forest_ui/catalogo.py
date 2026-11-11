"""Candidatos discretos de regression/casuisticas.py para el laboratorio."""

CATALOGO = {
    'catboost': ('CatBoost', {}),
    'gaussian_process': ('Gaussian Process', {}),
    'lightgbm': ('LightGBM', {}),
    'mlp': ('MLP', {}),
    'quantile': ('Regresión cuantílica', {}),
    'xgboost': ('XGBoost', {}),
    'random_forest': ('Random Forest', {}),
    'bayesian_ridge': ('Bayesian Ridge', {
        'alpha_1': [1e-6, 1e-4], 'alpha_2': [1e-6, 1e-4],
        'lambda_1': [1e-6, 1e-4], 'lambda_2': [1e-6, 1e-4],
        'max_iter': [300, 1000], 'tol': [1e-4, 1e-3],
        'alpha_init': [None, .1, 1., 10.], 'lambda_init': [None, .1, 1., 10.],
        'compute_score': [False, True], 'fit_intercept': [True, False], 'copy_X': [True, False],
    }),
    'ransac': ('RANSAC', {'min_samples': [None, .5, .75], 'max_trials': [50, 100, 200]}),
    'huber': ('Huber', {'epsilon': [1.1, 1.35, 1.5, 1.75], 'alpha': [.0001, .001, .01]}),
    'theil_sen': ('Theil-Sen', {'max_subpopulation': [100, 250, 500]}),
    'decision_tree': ('Árbol de decisión', {'max_depth': [None, 3, 5, 10], 'min_samples_leaf': [1, 2, 5]}),
    'gradient_boosting': ('Gradient Boosting', {'n_estimators': [50, 100], 'learning_rate': [.03, .1], 'max_depth': [2, 3]}),
    'svr': ('SVR', {'kernel': ['linear', 'rbf'], 'C': [.1, 1., 10.], 'epsilon': [.01, .1, .5], 'gamma': ['scale', 'auto', .01]}),
    'knn': ('KNN', {'n_neighbors': [2, 3, 5, 7], 'weights': ['uniform', 'distance'], 'p': [1, 2]}),
}
