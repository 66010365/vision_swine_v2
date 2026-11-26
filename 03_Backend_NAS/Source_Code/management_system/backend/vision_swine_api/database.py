# ======================================================
# Import Libraries
# ======================================================

import os

import psycopg
from dotenv import load_dotenv


# ======================================================
# Environment Configuration
# ======================================================

load_dotenv(
    "/home/dev/vision-swine/management_system/backend/.env"
)


# ======================================================
# Database Configuration
# ======================================================

DATABASE_CONFIG = {
    "dbname": os.environ["DATABASE_NAME"],
    "user": os.environ["DATABASE_USER"],
    "password": os.environ["DATABASE_PASSWORD"],
    "host": os.environ["DATABASE_HOST"],
    "port": int(os.environ["DATABASE_PORT"]),
}


# ======================================================
# Database Connection
# ======================================================

def get_connection():
    """Create and return a PostgreSQL database connection."""
    return psycopg.connect(**DATABASE_CONFIG)
