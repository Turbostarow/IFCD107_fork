FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app --home /home/app app \
    && mkdir -p /app/runs \
    && chown -R app:app /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .
USER app

EXPOSE 5000
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:5000", "app:app"]
