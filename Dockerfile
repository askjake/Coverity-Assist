# Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for psycopg2 (DB-ready) + curl for debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps – includes DB libs so you're ready when RDS shows up
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    requests \
    pydantic \
    sqlalchemy \
    psycopg2-binary \
    python-dotenv

# Copy your app
COPY app.py /app/app.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
