# ======================================================
# Import Libraries
# ======================================================

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..database import get_connection


# ======================================================
# Router Configuration
# ======================================================

router = APIRouter()


# ======================================================
# Session Schema
# ======================================================

class SessionCreate(BaseModel):
    device_id: int = Field(gt=0)
    camera_id: int | None = Field(default=None, gt=0)
    source_type: Literal["LIVE", "VIDEO"]


# ======================================================
# Create Processing Session
# ======================================================

@router.post("/sessions", tags=["Sessions"])
def create_session(data: SessionCreate) -> dict:
    """Create a new AI processing session."""

    with get_connection() as conn:
        with conn.cursor() as cur:

            # Check device
            cur.execute(
                "SELECT device_id FROM devices WHERE device_id = %s",
                (data.device_id,)
            )

            if cur.fetchone() is None:
                raise HTTPException(
                    status_code=404,
                    detail="Device not found"
                )

            # Check camera (optional)
            if data.camera_id is not None:
                cur.execute(
                    "SELECT camera_id FROM cameras WHERE camera_id = %s",
                    (data.camera_id,)
                )

                if cur.fetchone() is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Camera not found"
                    )

            # Insert new session
            cur.execute(
                """
                INSERT INTO processing_sessions (
                    device_id,
                    camera_id,
                    source_type
                )
                VALUES (%s, %s, %s)
                RETURNING
                    session_id,
                    device_id,
                    camera_id,
                    source_type,
                    started_at,
                    ended_at
                """,
                (
                    data.device_id,
                    data.camera_id,
                    data.source_type
                )
            )

            columns = [
                description.name
                for description in cur.description
            ]

            row = cur.fetchone()

    return dict(zip(columns, row))


# ======================================================
# Get Processing Sessions
# ======================================================

@router.get("/sessions", tags=["Sessions"])
def get_sessions() -> list[dict]:
    """Retrieve the latest 100 processing sessions with device names."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ps.session_id,
                    ps.device_id,
                    d.device_name,
                    ps.camera_id,
                    ps.source_type,
                    ps.started_at,
                    ps.ended_at
                FROM public.processing_sessions AS ps
                JOIN public.devices AS d
                    ON ps.device_id = d.device_id
                ORDER BY ps.started_at DESC
                LIMIT 100
                """
            )

            columns = [
                description.name
                for description in cur.description
            ]

            rows = cur.fetchall()

    return [
        dict(zip(columns, row))
        for row in rows
    ]


# ======================================================
# End Processing Session
# ======================================================

@router.patch("/sessions/{session_id}/end", tags=["Sessions"])
def end_session(session_id: UUID) -> dict:
    """Mark an active AI processing session as finished."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.processing_sessions
                SET ended_at = NOW()
                WHERE session_id = %s
                  AND ended_at IS NULL
                RETURNING
                    session_id,
                    device_id,
                    camera_id,
                    source_type,
                    started_at,
                    ended_at
                """,
                (session_id,)
            )

            columns = [
                description.name
                for description in cur.description
            ]
            row = cur.fetchone()

            if row is None:
                cur.execute(
                    """
                    SELECT ended_at
                    FROM public.processing_sessions
                    WHERE session_id = %s
                    """,
                    (session_id,)
                )
                existing = cur.fetchone()

                if existing is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Processing session not found"
                    )

                raise HTTPException(
                    status_code=409,
                    detail="Processing session has already ended"
                )

    return dict(zip(columns, row))
