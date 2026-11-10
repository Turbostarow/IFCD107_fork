CREATE TABLE IF NOT EXISTS regression_runs (
    run_id TEXT PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reference TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    model_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    finished_at TIMESTAMPTZ,
    error TEXT
);

CREATE TABLE IF NOT EXISTS regression_run_files (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES regression_runs(run_id) ON DELETE CASCADE,
    file_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, file_path)
);

CREATE TABLE IF NOT EXISTS regression_run_metrics (
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
);
