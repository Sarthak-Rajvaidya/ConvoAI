
"""
Fast audio acquisition.

Fast path (the default, and what makes this ~10x faster than the previous
implementation):

    YouTube / file  -->  compressed audio, untouched  -->  Groq transcription

The old pipeline always transcoded everything to mono 16kHz WAV and split it
into fixed 10-minute WAV chunks before transcription even started. WAV is
~10x larger than the compressed source and none of that preprocessing is
needed: Groq's transcription API accepts compressed audio (m4a/webm/mp3/ogg)
directly, so we skip re-encoding whenever possible.

Large-file path (only when needed):

    Large audio -> ffmpeg stream-copy segmentation -> compressed chunks
                -> parallel transcription -> merged transcript

-c copy segmentation re-packages the existing compressed stream into smaller
files without decoding/re-encoding, so it's fast even for long recordings.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import yt_dlp

from config.settings import Settings
from utils.errors import AudioDownloadError, ValidationError
from utils.logging import get_logger
from utils.validation import (
    GROQ_NATIVE_EXTENSIONS,
    is_youtube_url,
    safe_upload_filename,
    validate_upload_extension,
    validate_upload_size,
)

logger = get_logger(__name__)

WORK_ROOT = Path("tmp_audio")


class AcquiredAudio:
    def __init__(
        self,
        path: Path,
        label: str,
        duration_seconds: float | None,
        work_dir: Path,
    ):
        self.path = path
        self.label = label
        self.duration_seconds = duration_seconds
        self.work_dir = work_dir


def new_work_dir() -> Path:
    WORK_ROOT.mkdir(exist_ok=True)

    work_dir = WORK_ROOT / uuid.uuid4().hex[:12]
    work_dir.mkdir(parents=True, exist_ok=True)

    return work_dir


def _ffprobe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

        return float(out.stdout.strip())

    except Exception as exc:  # noqa: BLE001 - best effort, never fatal
        logger.debug(
            "ffprobe duration failed for %s: %s",
            path,
            exc,
        )
        return None


def download_youtube_audio(
    url: str,
    work_dir: Path,
) -> AcquiredAudio:
    """
    Download audio from YouTube.

    Uses the Android player client because normal YouTube extraction can be
    blocked from server-side/datacenter environments such as Render.

    The Android client may return an MP4 containing both video and audio.
    If that happens, the existing FFmpeg conversion path extracts the audio
    into a compact MP3 before sending it to Groq.
    """

    if not is_youtube_url(url):
        raise ValidationError(
            "That doesn't look like a supported YouTube URL."
        )

    outtmpl = str(work_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        # Android currently provides a downloadable combined media format
        # in server-side environments where the default client may be blocked.
        "format": "bestaudio/best",

        "outtmpl": outtmpl,

        # Use YouTube's Android player client.
        "extractor_args": {
            "youtube": {
                "player_client": ["android"],
            },
        },

        # Keep yt-dlp quiet inside the Streamlit application.
        "quiet": True,
        "no_warnings": True,

        # Never download an entire playlist when a playlist URL is supplied.
        "noplaylist": True,

        # Keep filenames safe across Windows/Linux/Render.
        "restrictfilenames": True,

        # Network reliability.
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,

        # Do not perform unnecessary post-processing.
        # FFmpeg conversion is handled below only when required.
        "postprocessors": [],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                url,
                download=True,
            )

            downloaded_path = Path(
                ydl.prepare_filename(info)
            )

    except yt_dlp.utils.DownloadError as exc:
        logger.error(
            "YouTube download failed: %s",
            exc,
        )

        raise AudioDownloadError(
            "Couldn't download that YouTube video. It may be private, "
            "age-restricted, region-locked, unavailable, or blocked by "
            "YouTube's current extraction restrictions.",
            detail=str(exc),
        ) from exc

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Unexpected YouTube download error: %s",
            exc,
        )

        raise AudioDownloadError(
            detail=str(exc),
        ) from exc

    if not downloaded_path.exists():
        raise AudioDownloadError(
            "Download finished but the audio file is missing."
        )

    title = str(
        info.get("title") or url
    )

    duration = info.get("duration")

    ext = downloaded_path.suffix.lstrip(".").lower()

    # Android may provide MP4 containing video + audio.
    # Convert unsupported containers to compact MP3 so Groq receives
    # an audio-only file.
    if ext not in GROQ_NATIVE_EXTENSIONS:
        downloaded_path = _transcode_to_mp3(
            downloaded_path
        )

    return AcquiredAudio(
        downloaded_path,
        title,
        duration,
        work_dir,
    )


def acquire_uploaded_file(
    tmp_uploaded_path: Path,
    original_filename: str,
    size_bytes: int,
    settings: Settings,
) -> AcquiredAudio:
    """
    Acquire a user-uploaded audio/video file.

    Native Groq audio formats are kept untouched.
    Other formats are converted to compact MP3.
    """

    safe_name = safe_upload_filename(
        original_filename
    )

    ext = validate_upload_extension(
        safe_name
    )

    validate_upload_size(
        size_bytes,
        settings.max_upload_mb,
    )

    work_dir = new_work_dir()

    dest = work_dir / safe_name

    shutil.copyfile(
        tmp_uploaded_path,
        dest,
    )

    if ext not in GROQ_NATIVE_EXTENSIONS:
        dest = _transcode_to_mp3(
            dest
        )

    duration = _ffprobe_duration(
        dest
    )

    return AcquiredAudio(
        dest,
        safe_name,
        duration,
        work_dir,
    )


def _transcode_to_mp3(
    path: Path,
) -> Path:
    """
    Only used for formats Groq doesn't accept natively
    (for example .mov/.mkv video containers).

    Extracts audio to a compact 96 kbps MP3.
    """

    out_path = path.with_suffix(
        ".mp3"
    )

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vn",
                "-acodec",
                "libmp3lame",
                "-b:a",
                "96k",
                str(out_path),
            ],
            capture_output=True,
            timeout=600,
            check=True,
        )

    except subprocess.CalledProcessError as exc:
        raise AudioDownloadError(
            "Couldn't process that audio/video file.",
            detail=(
                exc.stderr.decode(
                    "utf-8",
                    "ignore",
                )[-500:]
                if exc.stderr
                else str(exc)
            ),
        ) from exc

    return out_path


def segment_audio(
    audio: AcquiredAudio,
    chunk_seconds: int,
) -> list[tuple[Path, float]]:
    """
    Split a large audio file into compressed chunks using stream-copy
    (no re-encode) so long recordings segment in seconds, not minutes.

    Returns:
        [(chunk_path, start_offset_seconds), ...]
    """

    ext = (
        audio.path.suffix.lstrip(".")
        or "mp3"
    )

    out_pattern = (
        audio.work_dir
        / f"chunk_%03d.{ext}"
    )

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio.path),
                "-f",
                "segment",
                "-segment_time",
                str(chunk_seconds),
                "-c",
                "copy",
                "-reset_timestamps",
                "1",
                str(out_pattern),
            ],
            capture_output=True,
            timeout=900,
            check=True,
        )

    except subprocess.CalledProcessError:
        # Stream-copy can fail for some containers.
        # Fall back to real MP3 re-encoding while keeping the
        # chunks parallelizable afterwards.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio.path),
                "-f",
                "segment",
                "-segment_time",
                str(chunk_seconds),
                "-acodec",
                "libmp3lame",
                "-b:a",
                "96k",
                "-reset_timestamps",
                "1",
                str(
                    audio.work_dir
                    / "chunk_%03d.mp3"
                ),
            ],
            capture_output=True,
            timeout=1200,
            check=True,
        )

    chunks = sorted(
        audio.work_dir.glob(
            "chunk_*.*"
        )
    )

    if not chunks:
        raise AudioDownloadError(
            "Audio segmentation produced no chunks."
        )

    result: list[
        tuple[Path, float]
    ] = []

    offset = 0.0

    for chunk_path in chunks:
        result.append(
            (
                chunk_path,
                offset,
            )
        )

        offset += chunk_seconds

    return result


def needs_segmentation(
    audio: AcquiredAudio,
    max_segment_mb: int,
) -> bool:
    """
    Determine whether an audio file exceeds the configured
    maximum segment size.
    """

    size_mb = (
        audio.path.stat().st_size
        / 1_000_000
    )

    return size_mb > max_segment_mb


def cleanup(
    audio: AcquiredAudio,
) -> None:
    """
    Remove temporary audio files after processing.
    """

    try:
        shutil.rmtree(
            audio.work_dir,
            ignore_errors=True,
        )

    except Exception as exc:
        logger.debug(
            "Cleanup failed for %s: %s",
            audio.work_dir,
            exc,
        )
