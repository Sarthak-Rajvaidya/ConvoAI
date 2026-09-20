"""Tests for services/transcription/groq_provider.py -- success, empty
transcript, and provider failure/retry, with the Groq client mocked out."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.errors import ConfigurationError, TranscriptionError


def _make_provider():
    from services.transcription.groq_provider import GroqWhisperProvider
    with patch("services.transcription.groq_provider.Groq"):
        provider = GroqWhisperProvider(api_key="test-key", model="whisper-large-v3-turbo")
    return provider


def test_missing_api_key_raises_configuration_error():
    from services.transcription.groq_provider import GroqWhisperProvider
    with pytest.raises(ConfigurationError):
        GroqWhisperProvider(api_key="", model="whisper-large-v3-turbo")


def test_successful_transcription(tmp_path):
    provider = _make_provider()
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"fake audio bytes")

    response = MagicMock()
    response.text = "Hello team, let's begin."
    response.language = "en"
    response.duration = 12.5
    response.segments = [{"start": 0.0, "end": 2.0, "text": "Hello team,"}]
    provider._client.audio.transcriptions.create.return_value = response

    transcript = provider.transcribe(audio_file)

    assert transcript.text == "Hello team, let's begin."
    assert transcript.language == "en"
    assert transcript.provider == "groq"
    assert len(transcript.segments) == 1


def test_empty_transcript_is_returned_not_raised(tmp_path):
    provider = _make_provider()
    audio_file = tmp_path / "silence.mp3"
    audio_file.write_bytes(b"fake")

    response = MagicMock()
    response.text = ""
    response.language = "en"
    response.duration = 5.0
    response.segments = []
    provider._client.audio.transcriptions.create.return_value = response

    transcript = provider.transcribe(audio_file)
    assert transcript.text == ""
    # It's the pipeline's job to raise on an empty transcript, not the provider.


def test_provider_failure_retries_then_raises(tmp_path):
    provider = _make_provider()
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"fake")

    provider._client.audio.transcriptions.create.side_effect = RuntimeError("service unavailable")

    with pytest.raises(TranscriptionError):
        provider.transcribe(audio_file)

    assert provider._client.audio.transcriptions.create.call_count == 2  # bounded retry
