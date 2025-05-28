# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# This ensures that this layer is cached if requirements.txt doesn't change
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the flask_app directory into the container at /app
COPY ./flask_app ./flask_app

# Copy the tests directory into the container at /app (optional, but good practice for CI/CD)
COPY ./tests ./tests

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variables
ENV FLASK_APP=flask_app.app
# FLASK_RUN_HOST is set in the CMD, but can also be set here if preferred for flask run
# ENV FLASK_RUN_HOST=0.0.0.0

# Run the application when the container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
