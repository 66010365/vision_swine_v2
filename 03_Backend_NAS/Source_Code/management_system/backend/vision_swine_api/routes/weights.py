# ======================================================
# Import Libraries
# ======================================================

from decimal import Decimal
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..database import get_connection


# ======================================================
# Router Configuration
# ======================================================

router = APIRouter()


# ======================================================
# Weight Schema
# ======================================================

class WeightCreate(BaseModel):
    session_id: UUID
    track_id: int = Field(gt=0)
    weight_kg: Decimal = Field(
        gt=0,
        max_digits=7,
        decimal_places=2
    )
    frame_number: int = Field(ge=0)


# ======================================================
# Create Weight Estimation
# ======================================================

@router.post("/weights", tags=["Weights"])
def create_weight(data: WeightCreate) -> dict:
    """Save a pig weight estimation."""

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

            # Save weight estimation
            cur.execute(
                """
                INSERT INTO weight_estimations (
                    session_id,
                    track_id,
                    weight_kg,
                    frame_number
                )
                VALUES (%s, %s, %s, %s)
                RETURNING
                    estimation_id,
                    session_id,
                    track_id,
                    weight_kg,
                    frame_number,
                    estimated_at
                """,
                (
                    data.session_id,
                    data.track_id,
                    data.weight_kg,
                    data.frame_number
                )
            )

            columns = [
                description.name
                for description in cur.description
            ]

            row = cur.fetchone()

    return dict(zip(columns, row))


# ======================================================
# Get Weight Estimations
# ======================================================

@router.get("/weights", tags=["Weights"])
def get_weights() -> list[dict]:
    """Retrieve the latest 100 weight estimations."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    estimation_id,
                    session_id,
                    track_id,
                    weight_kg,
                    frame_number,
                    estimated_at
                FROM public.weight_estimations
                ORDER BY estimated_at DESC, estimation_id DESC
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
