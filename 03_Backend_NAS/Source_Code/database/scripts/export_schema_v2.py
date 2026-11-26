
# ======================================================
# Vision Swine V2 - Export PostgreSQL Schema
# ======================================================

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv


# ======================================================
# Project Configuration
# ======================================================

PROJECT_DIR = Path("/home/dev/vision-swine")

ENV_FILE = (
    PROJECT_DIR
    / "management_system"
    / "backend"
    / ".env"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "database"
    / "sql"
    / "schema_v2_current.sql"
)


# ======================================================
# Load Database Configuration
# ======================================================

load_dotenv(ENV_FILE)

database_name = os.environ["DATABASE_NAME"]
database_user = os.environ["DATABASE_USER"]
database_host = os.environ["DATABASE_HOST"]
database_port = os.environ["DATABASE_PORT"]

env = os.environ.copy()
env["PGPASSWORD"] = os.environ["DATABASE_PASSWORD"]


# ======================================================
# Export Database Schema
# ======================================================

def export_schema():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    command = [
        "pg_dump",
        "-h", database_host,
        "-p", database_port,
        "-U", database_user,
        "-d", database_name,
        "--schema-only",
        "--no-owner",
        "--no-privileges",
        "-f", str(OUTPUT_FILE)
    ]

    try:
        subprocess.run(
            command,
            env=env,
            check=True
        )

        print("Schema export successful")
        print(f"Database: {database_name}")
        print(f"Output: {OUTPUT_FILE}")

    except subprocess.CalledProcessError:
        print("Schema export failed")
        raise


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":
    export_schema()
