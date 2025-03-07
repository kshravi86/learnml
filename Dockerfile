# Use official Python image as a base
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy project files to the container
COPY . .

# Install dependencies
#RUN pip install --upgrade pip && pip install -r requirements.txt

# Run tests to verify the build (optional)
#RUN pytest --verbose || echo "Tests failed but continuing with the build"

# Command to run the application (Modify as needed)
#CMD ["python", "app.py"]
