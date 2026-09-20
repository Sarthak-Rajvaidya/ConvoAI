"""Tests for services/rag/chat.py -- grounded answers, no-context handling,
bounded history, and provider-failure behaviour, with retrieval and Groq
mocked out (no network calls)."""

from unittest.mock import MagicMock, patch

import pytest

from models.rag import ChatTurn, SourceChunk
from services.rag.chat import NO_CONTEXT_MESSAGE, MeetingChat
from services.rag.retriever import RetrievalResult
from utils.errors import LLMError


def _make_chat(top_k=5, history_turns=6):
    settings = MagicMock(
        groq_api_key="test-key", groq_chat_model="test-model", rag_top_k=top_k,
        chat_history_turns=history_turns,
    )
    with patch("services.rag.chat.Groq"):
        chat = MeetingChat(settings)
    return chat


def test_no_relevant_sources_returns_no_context_message():
    chat = _make_chat()
    chat.retrieve_sources = MagicMock(
        return_value=RetrievalResult(sources=[], used_fallback=True, fallback_reason="none found")
    )

    stream, sources, used_fallback, reason = chat.stream_answer("random question", "meeting-1", [])

    assert "".join(stream) == NO_CONTEXT_MESSAGE
    assert sources == []


def test_normal_question_streams_grounded_answer():
    chat = _make_chat()
    fake_sources = [SourceChunk(chunk_id=0, text="We decided to deploy on Friday.", start=10, score=0.8)]
    chat.retrieve_sources = MagicMock(
        return_value=RetrievalResult(sources=fake_sources, used_fallback=False)
    )

    def fake_stream():
        for token in ["The ", "team ", "decided ", "Friday."]:
            chunk = MagicMock()
            chunk.choices = [MagicMock(delta=MagicMock(content=token))]
            yield chunk

    chat._client.chat.completions.create.return_value = fake_stream()

    stream, sources, used_fallback, _ = chat.stream_answer("When are we deploying?", "meeting-1", [])
    answer = "".join(stream)

    assert answer == "The team decided Friday."
    assert sources == fake_sources
    assert used_fallback is False


def test_follow_up_question_includes_bounded_history():
    chat = _make_chat(history_turns=2)
    fake_sources = [SourceChunk(chunk_id=0, text="Rahul owns deployment.", score=0.5)]
    chat.retrieve_sources = MagicMock(
        return_value=RetrievalResult(sources=fake_sources, used_fallback=False)
    )

    captured = {}

    def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content="Rahul."))]
        return iter([chunk])

    chat._client.chat.completions.create.side_effect = fake_create

    history = [
        ChatTurn(role="user", content="What did the team decide?"),
        ChatTurn(role="assistant", content="They decided to deploy Friday."),
        ChatTurn(role="user", content="older turn"),
        ChatTurn(role="assistant", content="older answer"),
    ]
    stream, *_ = chat.stream_answer("Who was responsible for it?", "meeting-1", history)
    "".join(stream)

    # Only the last 2 history turns should be included, plus system + user.
    roles = [m["role"] for m in captured["messages"]]
    assert roles.count("user") == 2  # 1 from bounded history + the new question
    assert roles.count("assistant") == 1


def test_provider_failure_raises_llm_error():
    chat = _make_chat()
    fake_sources = [SourceChunk(chunk_id=0, text="Some context.", score=0.5)]
    chat.retrieve_sources = MagicMock(
        return_value=RetrievalResult(sources=fake_sources, used_fallback=False)
    )
    chat._client.chat.completions.create.side_effect = RuntimeError("Groq is down")

    stream, *_ = chat.stream_answer("A question", "meeting-1", [])
    with pytest.raises(LLMError):
        list(stream)
