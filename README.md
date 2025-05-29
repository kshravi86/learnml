# 🚀 FastAPI PostgreSQL CRUD Application

## 📝 Overview

This project is a RESTful API built with FastAPI and PostgreSQL, demonstrating Create, Read, Update, and Delete (CRUD) operations for "Items". FastAPI provides high performance and automatic interactive API documentation.

You can access the interactive API documentation (Swagger UI) at `/docs` and ReDoc at `/redoc` when the application is running.

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
    git checkout feat/fastapi-crud # Ensure you are on the correct branch
    ```
    (Replace `<repository-url>` and `<project-directory>` accordingly)

2.  🌿 **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  📦 **Install dependencies:**
    Make sure your `requirements.txt` reflects FastAPI dependencies.
    ```bash
    pip install -r requirements.txt
    ```

4.  💾 **Database Setup:**
    *   Ensure your PostgreSQL server is running and you have access to it.
    *   Create a new database (e.g., `fastapidb`) and a user/role with appropriate permissions.
    *   ⚠️ **Update `SQLALCHEMY_DATABASE_URL` in `fastapi_app/database.py`** with your actual PostgreSQL username, password, host, and database name. For example:
        ```python
        SQLALCHEMY_DATABASE_URL = "postgresql://youruser:yourpassword@localhost/fastapidb"
        ```
    *   📜 Run the database setup script to create the necessary tables:
        ```bash
        python fastapi_app/database_setup.py
        ```

## ▶️ Running the Application

### 🖥️ Without Docker:

1.  Ensure your virtual environment is activated.
2.  Run the Uvicorn server (from the project root directory):
    ```bash
    uvicorn fastapi_app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    The `--reload` flag enables auto-reloading for development.
3.  Access the API at `http://127.0.0.1:8000`.
4.  Interactive API documentation:
    *   Swagger UI: 🔗 `http://127.0.0.1:8000/docs`
    *   ReDoc: 🔗 `http://127.0.0.1:8000/redoc`

### 🐳 With Docker:

1.  **Build the Docker image:**
    ```bash
    docker build -t fastapi-crud-app .
    ```
2.  **Run the Docker container:**
    *   Ensure your PostgreSQL database is accessible from within the Docker container. This might mean updating `SQLALCHEMY_DATABASE_URL` in `fastapi_app/database.py` to use your host machine's IP address as seen from Docker (e.g., `host.docker.internal` on Docker Desktop for Mac/Windows, or your machine's network IP) and then rebuilding the image. Alternatively, run PostgreSQL in another Docker container and use Docker networking.
    ```bash
    docker run -p 8000:8000 fastapi-crud-app
    ```
3.  Access the API and interactive docs as listed above (e.g., `http://127.0.0.1:8000/docs`).

## ✅ Running Tests

🧪 Ensure your virtual environment is activated and dependencies are installed.
From the project root directory:
```bash
python -m unittest discover -s tests -p "test_fastapi_app.py"
# If you have tests for both Flask and FastAPI and want to run all:
# python -m unittest discover -s tests -p "test_*.py"
```
This command will discover and run tests specifically for the FastAPI application.

## 📂 Application Structure

🌳
```
.
├── fastapi_app/              # Main FastAPI application package
│   ├── __init__.py           # Marks fastapi_app as a Python package
│   ├── main.py               # FastAPI app instance, path operations (routes)
│   ├── crud.py               # Reusable functions for database interactions (CRUD logic)
│   ├── models.py             # SQLAlchemy models (database table definitions)
│   ├── schemas.py            # Pydantic models (data validation and serialization)
│   ├── database.py           # Database engine, session, and Base setup for SQLAlchemy
│   └── database_setup.py     # Script to create database tables
├── tests/                    # Unit tests
│   ├── test_app.py           # (If Flask tests are still present on this branch)
│   └── test_fastapi_app.py   # Tests for the FastAPI application
├── .gitignore                # Files and directories to ignore for Git
├── Dockerfile                # For building the FastAPI Docker image
├── requirements.txt          # Python dependencies for FastAPI
└── README.md                 # This file (FastAPI version)
```
