# Python 3.10 matches the development environment
FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code and data
COPY . .

# Bake the ChromaDB vector index into the image at build time.
# This means no ingestion cold-start on first request — the index
# is ready the moment the container starts.
# GOOGLE_API_KEY must be passed as a build arg for the embedding calls.
ARG GOOGLE_API_KEY
ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}
ENV PYTHONPATH=/app
RUN python scripts/ingest.py

EXPOSE 8000

# Serve on 0.0.0.0 so Render/Docker can reach the port
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
