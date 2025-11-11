FROM python:3.11-slim

WORKDIR /app

# Install runtime deps
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" requests

# Copy only what we need for the proxy
COPY app.py /app/

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
