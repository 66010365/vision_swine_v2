# ======================================================
# Import Libraries
# ======================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ======================================================
# Import Project Modules
# ======================================================

from .database import get_connection
from .routes.sessions import router as sessions_router
from .routes.weights import router as weights_router
from .routes.counting import router as counting_router
from .routes.videos import router as videos_router
from .routes.storage import router as storage_router


# ======================================================
# Application Configuration
# ======================================================

app = FastAPI(
    title="Vision Swine API",
    description="Backend API for the Vision Swine management system.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ======================================================
# CORS Configuration - React Dashboard v6
# ======================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://100.90.162.55:4175",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)


# ======================================================
# Health Check Routes
# ======================================================

@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Check whether the API service is running."""

    return {
        "status": "ok",
        "service": "vision-swine-api",
    }


# ======================================================
# Database Health Check Routes
# ======================================================

@app.get("/health/database", tags=["Health"])
def database_health() -> dict[str, str]:
    """Check the database connection and PostgreSQL version."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT current_database(), version();"
            )

            database_name, postgresql_version = cur.fetchone()

    return {
        "status": "ok",
        "database": database_name,
        "postgresql": postgresql_version,
    }


# ======================================================
# API Routers
# ======================================================

app.include_router(sessions_router)
app.include_router(weights_router)
app.include_router(counting_router)
app.include_router(videos_router)
app.include_router(storage_router)
