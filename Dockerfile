# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables to optimize Python within Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install pip updates
RUN pip install --no-cache-dir --upgrade pip

# Copy dependencies list and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server script and static web page components
COPY app.py /app/
COPY static/ /app/static/

# Copy the trained machine learning model artifacts
COPY scaler.pkl /app/
COPY rf_model.pkl /app/
COPY pca.pkl /app/
COPY kmeans.pkl /app/
COPY feature_names.pkl /app/

# Expose port (Cloud Run uses the PORT environment variable)
EXPOSE 8080

# Run the FastAPI server via Uvicorn
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
