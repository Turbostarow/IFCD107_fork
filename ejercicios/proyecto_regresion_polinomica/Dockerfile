FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app --home /home/app app \
    && mkdir -p /app/runs \
    && chown -R app:app /app

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev libpq5 \
    && apt-mark manual libpq5 \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=app:app . .
USER app

EXPOSE 5000
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:5000", "app:app"]
