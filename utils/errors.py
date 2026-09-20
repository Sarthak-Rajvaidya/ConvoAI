"""
utils/errors.py
===============================================================================
A small, deliberate error hierarchy.

Every service raises one of these instead of letting raw exceptions (network
errors, SDK exceptions, JSON errors) bubble up to the UI. Each carries a
short, human-readable `user_message` safe to display directly, while the
full technical detail stays in the log.
===============================================================================
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all application errors."""

    default_message = "Something went wrong."

    def __init__(self, user_message: str | None = None, *, detail: str = ""):
        self.user_message = user_message or self.default_message
        self.detail = detail
        super().__init__(f"{self.user_message} ({detail})" if detail else self.user_message)


class ConfigurationError(AppError):
    default_message = "A required setting or API key is missing."


class AudioDownloadError(AppError):
    default_message = "Couldn't download or read the audio/video source."


class TranscriptionError(AppError):
    default_message = "Speech-to-text transcription failed."


class AnalysisError(AppError):
    default_message = "AI meeting analysis failed."


class RAGIndexError(AppError):
    default_message = "Couldn't build the semantic search index for this meeting."


class RAGRetrievalError(AppError):
    default_message = "Couldn't search the meeting's knowledge base."


class LLMError(AppError):
    default_message = "The AI model didn't respond correctly."


class ValidationError(AppError):
    default_message = "That input isn't valid."
