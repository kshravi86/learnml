# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# Assumes requirements.txt is in the root of the build context
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the fastapi_app directory into the container at /app/fastapi_app
COPY ./fastapi_app ./fastapi_app

# Copy the tests directory into the container at /app/tests
# This is good practice for being able to run tests inside the container if needed
COPY ./tests ./tests

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the Uvicorn server when the container launches
# It will look for an 'app' instance in fastapi_app/main.py
CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
