# ======================================================
# Import Libraries
# ======================================================

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException


# ======================================================
# API Router Configuration
# ======================================================

router = APIRouter(
    prefix="/system",
    tags=["System"],
)


# ======================================================
# Storage Configuration
# ======================================================

VIDEO_STORAGE_DIR = Path(
    os.getenv(
        "VISION_SWINE_VIDEO_DIR",
        "/home/dev/vision-swine/storage/videos",
    )
)

BYTES_PER_GB = 1024 ** 3
BYTES_PER_MB = 1024 ** 2


# ======================================================
# Get NAS Disk Usage
# ======================================================

def get_disk_usage():
    """
    Read total, used, and free disk space from
    the filesystem containing the video directory.
    """

    usage = shutil.disk_usage(VIDEO_STORAGE_DIR)

    total = usage.total
    used = usage.used
    free = usage.free

    usage_percent = (
        (used / total) * 100
        if total > 0 else 0.0
    )

    return {
        "total_gb": round(total / BYTES_PER_GB, 2),
        "used_gb": round(used / BYTES_PER_GB, 2),
        "free_gb": round(free / BYTES_PER_GB, 2),
        "usage_percent": round(usage_percent, 1),
    }


# ======================================================
# Get Video Storage Usage
# ======================================================

def get_video_storage_usage():
    """
    Calculate allocated disk space used by files
    inside the Vision Swine video directory.
    """

    total_bytes = 0

    for root, dirs, files in os.walk(
        VIDEO_STORAGE_DIR,
        followlinks=False,
    ):
        for filename in files:
            file_path = Path(root) / filename

            if file_path.is_symlink():
                continue

            try:
                file_stat = file_path.stat()

                total_bytes += file_stat.st_blocks * 512

            except FileNotFoundError:
                continue

    return {
        "videos_used_gb": round(
            total_bytes / BYTES_PER_GB, 3
        ),
        "videos_used_mb": round(
            total_bytes / BYTES_PER_MB, 1
        ),
    }


# ======================================================
# Storage Information API
# ======================================================

@router.get("/storage")
def get_storage_information():
    """
    GET /system/storage

    Return NAS disk usage and Vision Swine
    video storage information.
    """

    if not VIDEO_STORAGE_DIR.is_dir():
        raise HTTPException(
            status_code=503,
            detail="Video storage directory is unavailable",
        )

    try:
        disk_usage = get_disk_usage()

        video_usage = get_video_storage_usage()

        return {
            **disk_usage,
            **video_usage,
        }

    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail="Unable to read NAS storage information",
        ) from exc
