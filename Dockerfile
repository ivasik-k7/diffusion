# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11.3
FROM python:${PYTHON_VERSION}-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run your Python script when the container launches
CMD ["python", "auto.py"]