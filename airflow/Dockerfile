
FROM apache/airflow:3.3.1-python3.11

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    pandas \
    psycopg2-binary \
    python-dotenv \
    openpyxl \
    sqlalchemy
