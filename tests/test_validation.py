"""Tests for utils/validation.py -- URL validation, filenames, sizes."""

import pytest

from utils.errors import ValidationError
from utils.validation import (
    is_youtube_url,
    safe_upload_filename,
    validate_source_url,
    validate_upload_extension,
    validate_upload_size,
)


def test_valid_youtube_url():
    assert is_youtube_url("https://www.youtube.com/watch?v=abc123")
    assert is_youtube_url("https://youtu.be/abc123")


def test_invalid_youtube_url():
    assert not is_youtube_url("https://vimeo.com/12345")
    assert not is_youtube_url("not a url")


def test_validate_source_url_raises_on_empty():
    with pytest.raises(ValidationError):
        validate_source_url("")


def test_validate_source_url_raises_on_non_youtube():
    with pytest.raises(ValidationError):
        validate_source_url("https://example.com/video")


def test_safe_upload_filename_strips_path_traversal():
    assert safe_upload_filename("../../etc/passwd") == "passwd"
    assert safe_upload_filename("/tmp/../../secret.mp3") == "secret.mp3"


def test_safe_upload_filename_sanitizes_special_chars():
    result = safe_upload_filename("my recording!@#.mp3")
    assert " " not in result
    assert "!" not in result


def test_validate_upload_extension_supported():
    assert validate_upload_extension("recording.mp3") == "mp3"


def test_validate_upload_extension_unsupported():
    with pytest.raises(ValidationError):
        validate_upload_extension("script.exe")


def test_validate_upload_size_within_limit():
    validate_upload_size(10_000_000, max_mb=50)  # should not raise


def test_validate_upload_size_over_limit():
    with pytest.raises(ValidationError):
        validate_upload_size(100_000_000, max_mb=50)
