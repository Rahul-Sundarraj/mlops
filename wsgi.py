"""WSGI entry point: `gunicorn wsgi:app` (Azure App Service startup command)."""
from src.web.app import create_app

app = create_app()
