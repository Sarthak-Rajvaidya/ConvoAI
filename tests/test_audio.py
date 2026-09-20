"""Tests for services/audio/processor.py input handling -- URL validation,
unsupported file types, and oversized uploads are all rejected before any
network call or disk write happens."""

import io
from pathlib import Path

import pytest

from config.settings import Settings
from services.audio import processor as audio_processor
from utils.errors import AudioDownloadError, ValidationError


def test_download_rejects_non_youtube_url(tmp_path):
    with pytest.raises(ValidationError):
        audio_processor.download_youtube_audio("https://example.com/video.mp4", tmp_path)


def test_acquire_uploaded_file_rejects_unsupported_extension(tmp_path):
    fake_file = tmp_path / "malware.exe"
    fake_file.write_bytes(b"not audio")

    settings = Settings(max_upload_mb=300)
    with pytest.raises(ValidationError):
        audio_processor.acquire_uploaded_file(fake_file, "malware.exe", fake_file.stat().st_size, settings)


def test_acquire_uploaded_file_rejects_oversized_file(tmp_path):
    fake_file = tmp_path / "big.mp3"
    fake_file.write_bytes(b"0" * 1000)

    settings = Settings(max_upload_mb=0)  # anything is "too big"
    with pytest.raises(ValidationError):
        audio_processor.acquire_uploaded_file(fake_file, "big.mp3", 50_000_000, settings)


def test_acquire_uploaded_file_sanitizes_path_traversal_filename(tmp_path):
    fake_file = tmp_path / "clip.mp3"
    fake_file.write_bytes(b"0" * 1000)

    settings = Settings(max_upload_mb=300)
    result = audio_processor.acquire_uploaded_file(
        fake_file, "../../etc/clip.mp3", fake_file.stat().st_size, settings
    )
    assert ".." not in str(result.path)
    audio_processor.cleanup(result)


def test_needs_segmentation_respects_size_threshold(tmp_path):
    small_file = tmp_path / "small.mp3"
    small_file.write_bytes(b"0" * 1000)
    audio = audio_processor.AcquiredAudio(small_file, "small", 10.0, tmp_path)

    assert audio_processor.needs_segmentation(audio, max_segment_mb=0) is True
    assert audio_processor.needs_segmentation(audio, max_segment_mb=100) is False
