# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /usr/src/app 

# Copy the requirements file into the container
# Assumes requirements.txt is in the root of the build context (where Dockerfile is)
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire Django project directory 'django_crud_project' 
# (which contains manage.py and the inner project config directory)
# into the working directory '/usr/src/app' as './django_crud_project'
COPY ./django_crud_project ./django_crud_project

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run Django's development server.
# manage.py is now at /usr/src/app/django_crud_project/manage.py
CMD ["python", "django_crud_project/manage.py", "runserver", "0.0.0.0:8000"]
