# Use Python base image
FROM python:3.8-slim-buster

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Set the working directory
WORKDIR /app

# Remove unnecessary packages
RUN apt-get purge -y --auto-remove build-essential curl && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache

# Set the startup command to run the application
CMD ["python", "app.py"]

