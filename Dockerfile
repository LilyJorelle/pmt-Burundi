# Start from a small official Python image (same version as on your PC).
FROM python:3.12-slim

# XGBoost needs this system library (OpenMP) to run inside a slim image.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the libraries first, so Docker can reuse this layer when only the code changes.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only what the service needs: the API code and the trained model. No data.
COPY api/ api/
COPY models/ models/

# The service listens on port 8000; 0.0.0.0 makes it reachable from outside the container.
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]