FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so editing code does not reinstall them.
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY . .

# Run as a normal user. /db holds the SQLite database (a named volume in
# docker-compose.yml); the other folders are where config.py writes data.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /db /app/data/store /app/ml/model_store /app/logs \
    && chown -R app:app /db /app
USER app

EXPOSE 8501

# Default: the self-contained demo (synthetic data, no credentials needed).
CMD ["streamlit", "run", "demo_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]