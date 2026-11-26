# ======================================================
# Import Libraries
# ======================================================

import os
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..database import get_connection


# ======================================================
# Router Configuration
# ======================================================

router = APIRouter()


# ======================================================
# Storage Configuration
# ======================================================

STORAGE_DIR = Path("/home/dev/vision-swine/storage")
VIDEO_DIR = STORAGE_DIR / "videos"

# Maximum upload size: 2 GB
MAX_VIDEO_SIZE = 2 * 1024 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


# ======================================================
# Upload AI Processed Video
# ======================================================

@router.post("/videos", tags=["Videos"], status_code=201)
def upload_video(
    session_id: UUID = Form(...),
    file: UploadFile = File(...)
) -> dict:
    """Upload an AI-processed MP4 video and save its metadata."""

    original_name = Path(file.filename or "").name

    if not original_name or not original_name.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=400,
            detail="Only MP4 files are allowed"
        )

    # Check processing session before writing the file
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT session_id
                FROM public.processing_sessions
                WHERE session_id = %s
                """,
                (session_id,)
            )

            if cur.fetchone() is None:
                raise HTTPException(
                    status_code=404,
                    detail="Processing session not found"
                )

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # Generate a unique storage filename
    stored_name = f"{uuid4().hex}.mp4"
    relative_path = f"videos/{stored_name}"

    final_path = VIDEO_DIR / stored_name
    temp_path = VIDEO_DIR / f"{stored_name}.uploading"

    file_size = 0

    try:
        # Save video in chunks, avoiding full-file memory loading
        with temp_path.open("xb") as output:
            while True:
                chunk = file.file.read(CHUNK_SIZE)

                if not chunk:
                    break

                file_size += len(chunk)

                if file_size > MAX_VIDEO_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the maximum upload size"
                    )

                output.write(chunk)

        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty"
            )

        # Move completed upload to its final path
        os.replace(temp_path, final_path)

        # Save video metadata to PostgreSQL
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.video_recordings (
                        session_id,
                        file_name,
                        file_path,
                        file_size_bytes,
                        status
                    )
                    VALUES (%s, %s, %s, %s, 'READY')
                    RETURNING
                        video_id,
                        session_id,
                        file_name,
                        file_path,
                        file_size_bytes,
                        status,
                        created_at
                    """,
                    (
                        session_id,
                        original_name,
                        relative_path,
                        file_size
                    )
                )

                columns = [
                    description.name
                    for description in cur.description
                ]

                row = cur.fetchone()

        return dict(zip(columns, row))

    except Exception:
        # Remove incomplete or unregistered files on failure
        temp_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise

    finally:
        file.file.close()


# ======================================================
# Get Video List
# ======================================================

@router.get("/videos", tags=["Videos"])
def get_videos() -> list[dict]:
    """Retrieve the latest 100 recorded videos."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    video_id,
                    session_id,
                    file_name,
                    file_path,
                    file_size_bytes,
                    status,
                    created_at
                FROM public.video_recordings
                ORDER BY created_at DESC
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
# Play Recorded Video
# ======================================================

@router.get("/videos/{video_id}/file", tags=["Videos"])
def get_video_file(video_id: int):
    """Serve an AI-processed MP4 video by database video ID."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT file_path, status
                FROM public.video_recordings
                WHERE video_id = %s
                """,
                (video_id,)
            )

            row = cur.fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Video not found"
        )

    relative_path, status = row

    if status != "READY":
        raise HTTPException(
            status_code=409,
            detail="Video is not ready"
        )

    # Only serve files under storage/videos, never arbitrary server paths.
    video_root = VIDEO_DIR.resolve()
    video_path = (STORAGE_DIR / relative_path).resolve()

    if not video_path.is_relative_to(video_root):
        raise HTTPException(
            status_code=400,
            detail="Invalid video path"
        )

    if not video_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Video file is missing from storage"
        )

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        content_disposition_type="inline"
    )
