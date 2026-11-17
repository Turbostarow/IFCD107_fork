"""Persistencia PostgreSQL de ejecuciones de las casuísticas de regresión."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any

import psycopg2


SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS regression_runs (
        run_id TEXT PRIMARY KEY,
        started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        reference TEXT NOT NULL,
        algorithm TEXT NOT NULL,
        model_name TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'running',
        finished_at TIMESTAMPTZ,
        error TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS regression_run_files (
        id BIGSERIAL PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES regression_runs(run_id) ON DELETE CASCADE,
        file_type TEXT NOT NULL,
        file_path TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (run_id, file_path)
    )""",
    """CREATE TABLE IF NOT EXISTS regression_run_metrics (
        id BIGSERIAL PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES regression_runs(run_id) ON DELETE CASCADE,
        search_name TEXT NOT NULL,
        cv_rmse DOUBLE PRECISION,
        test_rmse DOUBLE PRECISION,
        test_r2 DOUBLE PRECISION,
        parameters_text TEXT NOT NULL,
        metrics_text TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (run_id, search_name)
    )""",
    """CREATE TABLE IF NOT EXISTS regression_run_logs (
            id BIGSERIAL PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES regression_runs(run_id) ON DELETE CASCADE,
            log_type TEXT NOT NULL,
            file_path TEXT ,
            logtext TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""",
)


def is_configured() -> bool:
    """Indica si hay conexión PostgreSQL configurada en el entorno."""
    return bool(os.environ.get("PGHOST"))


def _connect():
    """Abre una conexión usando variables estándar PG* de libpq."""
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "regression_results"),
        user=os.environ.get("PGUSER", "regression"),
        password=os.environ.get("PGPASSWORD", ""),
        connect_timeout=int(os.environ.get("PGCONNECT_TIMEOUT", "5")),
    )


@contextmanager
def _managed_connection():
    """Confirma o revierte la transacción y siempre libera la conexión."""
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def ensure_schema() -> bool:
    """Crea la estructura si no existe; útil también con volúmenes ya creados."""
    if not is_configured():
        return False
    with _managed_connection() as connection:
        with connection.cursor() as cursor:
            for statement in SCHEMA_STATEMENTS:
                cursor.execute(statement)
    return True


def register_run(run_id: str, reference: str, algorithm: str, 
                 model_name: str) -> bool:
    """Guarda el identificador, la fecha y el modelo antes de entrenar."""
    if not is_configured():
        return False
    ensure_schema()
    with _managed_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO regression_runs
                   (run_id, reference, algorithm, model_name, status)
                   VALUES (%s, %s, %s, %s, 'running')
                   ON CONFLICT (run_id) DO NOTHING""",
                (run_id, reference, algorithm, model_name),
            )
    return True

def register_log(run_id: str, log_type: str, 
                 logtext: str, file_path: str = "") -> bool:
    """Guarda run_id TEXT NOT NULL REFERENCES regression_runs(run_id) ON DELETE CASCADE,
                log_type TEXT NOT NULL,
                file_path TEXT ,
                logtext TEXT NOT NULL,"""
    if not is_configured():
        return False
    with _managed_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO regression_run_logs
                   (run_id, log_type, file_path, logtext)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (run_id) DO NOTHING""",
                (run_id, log_type,file_path , logtext),
            )
    return True

def get_run_model_path(run_id: str, root: str = "") -> str | None:
    """Devuelve la ruta completa del fichero .joblib asociado a una ejecución, o None si no existe."""
    if not is_configured():
        return None
    ensure_schema()
    file_type='model'
    with _managed_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                 """SELECT file_path 
                   FROM regression_run_files 
                   WHERE run_id = %s AND file_type = 'model'""",
                (run_id,),
            )
            row = cursor.fetchone()
    print('get_run_model_path')
    print(row)
    if row is None:
        return None
    print('get_run_model_path')
    print(os.path.join(root, row[0]) if root else row[0])
    return os.path.join(root, row[0]) if root else row[0]


def mark_run_failed(run_id: str, error: str) -> bool:
    """Marca como fallida una ejecución que no terminó correctamente."""
    if not is_configured():
        return False
    ensure_schema()
    with _managed_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """UPDATE regression_runs
                   SET status = 'failed', finished_at = NOW(), error = %s
                   WHERE run_id = %s""",
                (str(error)[:2000], run_id),
            )
            cursor.close()
        connection.commit()
    return True


def save_casuistica_results(run_id: str, summary: dict[str, Any]) -> bool:
    """Guarda archivos generados y una métrica por cada estrategia de búsqueda."""
    if not is_configured():
        return False
    ensure_schema()

    artifacts = summary.get("artifacts", {})
    comparisons = summary.get("comparison_metrics", {})
    searches = summary.get("search_comparison", {})
    winner = summary.get("winner")

    with _managed_connection() as connection:
        with connection.cursor() as cursor:
            for file_type, file_path in artifacts.items():
                cursor.execute(
                    """INSERT INTO regression_run_files (run_id, file_type, file_path)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (run_id, file_path)
                       DO UPDATE SET file_type = EXCLUDED.file_type""",
                    (run_id, str(file_type), str(file_path)),
                )

            for search_name in ("GridSearchCV", "RandomizedSearchCV"):
                search = searches.get(search_name, {})
                test_metrics = comparisons.get(search_name, {})
                parameters = search.get("best_parameters", {})
                metrics = {
                    "winner": winner == search_name,
                    "best_cv_rmse": search.get("best_cv_rmse"),
                    "test": test_metrics,
                    "overall_metrics": summary.get("metrics", {}),
                    "cv_folds": summary.get("cv_folds"),
                }
                cursor.execute(
                    """INSERT INTO regression_run_metrics
                       (run_id, search_name, cv_rmse, test_rmse, test_r2,
                        parameters_text, metrics_text)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (run_id, search_name) DO UPDATE SET
                         cv_rmse = EXCLUDED.cv_rmse,
                         test_rmse = EXCLUDED.test_rmse,
                         test_r2 = EXCLUDED.test_r2,
                         parameters_text = EXCLUDED.parameters_text,
                         metrics_text = EXCLUDED.metrics_text""",
                    (
                        run_id,
                        search_name,
                        search.get("best_cv_rmse"),
                        test_metrics.get("rmse"),
                        test_metrics.get("r2"),
                        json.dumps(parameters, ensure_ascii=False, default=str),
                        json.dumps(metrics, ensure_ascii=False, default=str),
                    ),
                )

            cursor.execute(
                """UPDATE regression_runs
                   SET status = 'completed', finished_at = NOW(), error = NULL
                   WHERE run_id = %s""",
                (run_id,),
            )
    return True
