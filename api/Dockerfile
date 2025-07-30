# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements if exists, then install
COPY api/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy source code
COPY api/ ./

# Copy model and scaler files
COPY models/random_forest_model.pkl models/scaler.pkl models/

# Add python path to include the current directory
ENV PYTHONPATH=/app

# Set entrypoint
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]