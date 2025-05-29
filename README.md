# 🏛️ Django Traditional CRUD Application

## 📝 Overview

This project is a web application built with Django, demonstrating traditional server-side CRUD (Create, Read, Update, Delete) operations for "Items". It utilizes Django's ORM for database interaction with PostgreSQL, Django Forms for data validation and input, and Django Templates for rendering HTML.

## 📋 Prerequisites

*   ✨ Python 3.8+
*   ✨ pip (Python package installer)
*   ✨ PostgreSQL server (running and accessible)
*   ✨ Docker (optional, for containerized deployment)

## 🛠️ Setup Instructions

1.  🔗 **Clone the repository and checkout the branch:**
    ```bash
    git clone <repository-url>
    cd <project-directory>
    git checkout feat/django-traditional-crud # Or the current branch name
    ```
    (Replace `<repository-url>` and `<project-directory>` accordingly)

2.  🌿 **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  📦 **Install dependencies:**
    Make sure your `requirements.txt` reflects Django dependencies.
    ```bash
    pip install -r requirements.txt
    ```

4.  ⚙️ **Configure Project Settings:**
    *   Navigate to `django_crud_project/` (the directory containing `manage.py`).

5.  💾 **Database Setup:**
    *   Ensure your PostgreSQL server is running and you have access to it.
    *   Create a new database (e.g., `djangocruddb`) and a user/role with appropriate permissions.
    *   ⚠️ **Update `DATABASES` setting in `django_crud_project/django_crud_project/settings.py`** with your actual PostgreSQL credentials (username, password, host, database name). For example:
        ```python
        DATABASES = {
            'default': {
                'ENGINE': 'django.db.backends.postgresql',
                'NAME': 'your_db_name',
                'USER': 'your_db_user',
                'PASSWORD': 'your_db_password',
                'HOST': 'localhost',
                'PORT': '5432',
            }
        }
        ```
    *   Run database migrations (from the directory containing `manage.py`):
        ```bash
        python manage.py makemigrations items_app
        python manage.py migrate
        ```

6.  👤 **Create a Superuser (for Admin Panel):** (Optional, but recommended)
    Run the following command from the directory containing `manage.py`:
    ```bash
    python manage.py createsuperuser
    ```
    Follow the prompts to create an admin user.

## ▶️ Running the Application

### 🖥️ Without Docker (Development Server):

1.  Ensure your virtual environment is activated and you are in the `django_crud_project` directory (the one containing `manage.py`).
2.  Run the development server:
    ```bash
    python manage.py runserver
    ```
3.  Access the application at: 🔗 `http://127.0.0.1:8000/items/`
4.  Access Django Admin: 🔗 `http://127.0.0.1:8000/admin/` (login with your superuser credentials)

### 🐳 With Docker:

1.  **Build the Docker image:**
    ```bash
    docker build -t django-crud-app .
    ```
2.  **Run the Docker container:**
    *   Ensure your PostgreSQL database is accessible from within the Docker container. This might mean updating the `DATABASES` setting in `django_crud_project/django_crud_project/settings.py` to use your host machine's IP address as seen from Docker (e.g., `host.docker.internal` on Docker Desktop for Mac/Windows, or your machine's network IP) and then rebuilding the image. Alternatively, run PostgreSQL in another Docker container and use Docker networking.
    *   You might also need to run migrations within the Docker container the first time or use an entrypoint script to handle this. For simplicity, this example assumes migrations are handled or the DB is already set up.
    ```bash
    docker run -p 8000:8000 django-crud-app
    ```
3.  Access the application as listed above (e.g., `http://127.0.0.1:8000/items/`).

## ✅ Running Tests

🧪 Ensure your virtual environment is activated and you are in the `django_crud_project` directory (the one containing `manage.py`).
```bash
python manage.py test items_app
# Or to run all tests discovered in the project:
# python manage.py test
```
This command will discover and run tests specifically for the `items_app` or all apps.

## 📂 Application Structure

🌳
```
.
├── django_crud_project/      # Root directory for the Django project.
│   ├── manage.py             # Django's command-line utility.
│   ├── django_crud_project/  # Inner Python package for project configurations.
│   │   ├── __init__.py
│   │   ├── settings.py       # Project settings (DB config, installed apps, etc.).
│   │   ├── urls.py           # Project-level URL routing.
│   │   ├── wsgi.py           # WSGI configuration.
│   │   └── asgi.py           # ASGI configuration.
│   ├── items_app/            # The application for handling items.
│   │   ├── __init__.py
│   │   ├── models.py         # Database models for items.
│   │   ├── views.py          # View functions for handling requests.
│   │   ├── forms.py          # Django forms for item creation/editing.
│   │   ├── urls.py           # URL routing for the items_app.
│   │   ├── templates/
│   │   │   └── items_app/    # HTML templates for the items_app.
│   │   │       ├── base.html
│   │   │       ├── item_list.html
│   │   │       ├── item_detail.html
│   │   │       ├── item_form.html
│   │   │       └── item_confirm_delete.html
│   │   ├── admin.py          # Admin site configuration for models.
│   │   ├── tests.py          # Unit tests for the app.
│   │   └── migrations/       # Database migration files.
│   │       └── __init__.py
├── Dockerfile                # For building the Django Docker image.
├── requirements.txt          # Python dependencies (Django, psycopg2-binary, etc.).
└── README.md                 # This file (Django Traditional Views version).
```
