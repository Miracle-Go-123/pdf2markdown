# Use Python 3.12 slim version for a lightweight image
FROM python:3.12-slim

# Install system dependencies (including Poppler)
RUN apt-get update && apt-get install -y poppler-utils

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the application port
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
