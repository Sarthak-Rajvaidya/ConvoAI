"""
utils/validation.py
===============================================================================
Small, dependency-free validation helpers used before any file touches disk
or any URL is handed to yt-dlp. Centralising this makes the security pass
(section 30 of the brief) auditable in one place.
===============================================================================
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from utils.errors import ValidationError

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "music.youtube.com",
}

SUPPORTED_UPLOAD_EXTENSIONS = {
    "mp3", "wav", "m4a", "mp4", "webm", "mov", "mkv", "aac", "ogg", "flac", "mpga", "mpeg",
}

# Formats Groq's transcription API accepts directly, without re-encoding.
GROQ_NATIVE_EXTENSIONS = {"flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"}

_YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")


def is_youtube_url(source: str) -> bool:
    try:
        parsed = urlparse(source.strip())
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    return parsed.netloc.lower() in YOUTUBE_HOSTS


def validate_source_url(source: str) -> str:
    source = (source or "").strip()
    if not source:
        raise ValidationError("Please paste a YouTube URL.")
    if not is_youtube_url(source):
        raise ValidationError(
            "That doesn't look like a supported YouTube URL. "
            "Paste a link like https://youtube.com/watch?v=..."
        )
    return source


def safe_upload_filename(filename: str) -> str:
    """Strip any path components to prevent path traversal via a crafted
    upload filename (e.g. '../../etc/passwd')."""
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9_.\-]", "_", name)
    if not name or name in (".", ".."):
        raise ValidationError("Invalid file name.")
    return name


def validate_upload_extension(filename: str) -> str:
    ext = Path(filename).suffix.lstrip(".").lower()
    if ext not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file type '.{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))}"
        )
    return ext


def validate_upload_size(size_bytes: int, max_mb: int) -> None:
    size_mb = size_bytes / 1_000_000
    if size_mb > max_mb:
        raise ValidationError(
            f"File is too large ({size_mb:.1f} MB). Maximum is {max_mb} MB."
        )
