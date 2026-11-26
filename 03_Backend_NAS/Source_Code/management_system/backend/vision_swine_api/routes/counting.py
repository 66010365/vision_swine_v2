# ======================================================
# Import Libraries
# ======================================================

from datetime import datetime, timezone
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
# Counting Event Schema
# ======================================================

class CountingEventCreate(BaseModel):
    session_id: UUID
    track_id: int = Field(gt=0)
    direction: Literal["IN", "OUT"]
    frame_number: int | None = Field(default=None, ge=0)
    event_time: datetime | None = None
    weight_estimation_id: int | None = Field(default=None, gt=0)


# ======================================================
# Create Counting Event
# ======================================================

@router.post("/counting-events", tags=["Counting"])
def create_counting_event(data: CountingEventCreate) -> dict:
    """Save an IN/OUT counting event."""

    event_time = data.event_time or datetime.now(timezone.utc)

    with get_connection() as conn:
        with conn.cursor() as cur:

            # Check processing session
            cur.execute(
                """
                SELECT session_id
                FROM processing_sessions
                WHERE session_id = %s
                """,
                (data.session_id,)
            )

            if cur.fetchone() is None:
                raise HTTPException(
                    status_code=404,
                    detail="Processing session not found"
                )

            # Check linked weight estimation, if provided
            if data.weight_estimation_id is not None:
                cur.execute(
                    """
                    SELECT estimation_id
                    FROM weight_estimations
                    WHERE estimation_id = %s
                      AND session_id = %s
                      AND track_id = %s
                    """,
                    (
                        data.weight_estimation_id,
                        data.session_id,
                        data.track_id
                    )
                )

                if cur.fetchone() is None:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Weight estimation does not match "
                            "the session and track ID"
                        )
                    )

            # Save counting event
            cur.execute(
                """
                INSERT INTO counting_events (
                    session_id,
                    track_id,
                    direction,
                    frame_number,
                    event_time,
                    weight_estimation_id
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING
                    event_id,
                    session_id,
                    track_id,
                    direction,
                    frame_number,
                    event_time,
                    weight_estimation_id
                """,
                (
                    data.session_id,
                    data.track_id,
                    data.direction,
                    data.frame_number,
                    event_time,
                    data.weight_estimation_id
                )
            )

            columns = [
                description.name
                for description in cur.description
            ]

            row = cur.fetchone()

    return dict(zip(columns, row))


# ======================================================
# Get Counting Events
# ======================================================

@router.get("/counting-events", tags=["Counting"])
def get_counting_events() -> list[dict]:
    """Retrieve the latest 100 IN/OUT counting events with linked weights."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ce.event_id,
                    ce.session_id,
                    ce.track_id,
                    ce.direction,
                    ce.frame_number,
                    ce.event_time,
                    ce.weight_estimation_id,
                    we.weight_kg
                FROM public.counting_events AS ce
                LEFT JOIN public.weight_estimations AS we
                    ON ce.weight_estimation_id = we.estimation_id
                   AND ce.session_id = we.session_id
                   AND ce.track_id = we.track_id
                ORDER BY ce.event_time DESC, ce.event_id DESC
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
