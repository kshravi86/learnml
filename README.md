# Flask PostgreSQL CRUD Application

## Overview

This is a simple web application built with Flask and PostgreSQL that demonstrates Create, Read, Update, and Delete (CRUD) operations. It provides both a JSON API and an HTML web interface for interacting with a database of "Items".

## Prerequisites

*   Python 3.8+
*   pip (Python package installer)
*   PostgreSQL server (running and accessible)
*   Docker (optional, for containerized deployment)

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
    (Replace `<repository-url>` with the actual URL of this repository)

2.  **Navigate to the project directory:**
    ```bash
    cd <project-directory>
    ```
    (Replace `<project-directory>` with the name of the cloned folder)

3.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Database Setup:**
    *   Ensure your PostgreSQL server is running and you have access to it.
    *   Create a new database (e.g., `mydatabase`) and a user/role (e.g., `user`) with appropriate permissions (e.g., ability to connect, create tables, CRUD operations on the database).
    *   **Crucially, update the `SQLALCHEMY_DATABASE_URI` in `flask_app/app.py`**. Open the file `flask_app/app.py` and modify the line:
        ```python
        app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@localhost/mydatabase'
        ```
        Replace `user`, `password`, `localhost`, and `mydatabase` with your actual PostgreSQL username, password, host, and database name.
    *   Run the database setup script to create the necessary tables:
        ```bash
        python flask_app/database_setup.py
        ```

## Running the Application

### Without Docker:

1.  Ensure your virtual environment is activated.
2.  Set the Flask application environment variable:
    ```bash
    export FLASK_APP=flask_app.app  # On Windows use: set FLASK_APP=flask_app.app
    ```
3.  Run the Flask development server:
    ```bash
    flask run
    ```
4.  Access the application:
    *   HTML UI: [http://127.0.0.1:5000/ui/items](http://127.0.0.1:5000/ui/items)
    *   JSON API: [http://127.0.0.1:5000/items](http://127.0.0.1:5000/items)

### With Docker:

1.  **Build the Docker image:**
    ```bash
    docker build -t flask-crud-app .
    ```
2.  **Run the Docker container:**
    *   Ensure your PostgreSQL database is accessible from within the Docker container. This might mean:
        *   If PostgreSQL is running on your host, you might need to change `localhost` in `SQLALCHEMY_DATABASE_URI` (inside `flask_app/app.py`, then rebuild the image) to your host machine's IP address as seen from Docker (e.g., `host.docker.internal` on Docker Desktop for Mac/Windows, or your machine's network IP).
        *   Alternatively, run PostgreSQL in another Docker container and use Docker networking.
    ```bash
    docker run -p 5000:5000 flask-crud-app
    ```
3.  Access the application as above (e.g., `http://127.0.0.1:5000/ui/items`).

## Running Tests

Ensure your virtual environment is activated and dependencies are installed.

```bash
python -m unittest discover -s tests -p "test_*.py"
```
This command will discover and run all tests located in the `tests` directory.

## Application Structure

```
.
├── flask_app/                # Main application package
│   ├── app.py                # Flask application, routes, DB initialization
│   ├── models.py             # SQLAlchemy database models
│   ├── database_setup.py     # Script to create database tables
│   ├── static/               # Static files (CSS, JavaScript)
│   │   └── style.css
│   ├── templates/            # HTML templates
│   │   ├── index.html
│   │   ├── item_detail.html
│   │   └── item_form.html
│   └── .gitkeep              # (Placeholder if static/templates were empty)
├── tests/                    # Unit tests
│   └── test_app.py
├── .gitignore                # Files and directories to ignore for Git
├── Dockerfile                # For building the Docker image
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```
