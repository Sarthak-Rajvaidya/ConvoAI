"""
services/rag/chat.py
===============================================================================
The RAG chatbot: retrieval -> grounded prompt -> streaming Groq completion
-> sources. Bounded conversation memory lets follow-up questions ("who was
responsible for it?") resolve pronouns against the last few turns without
sending unlimited history.
===============================================================================
"""

from __future__ import annotations

from collections.abc import Iterator

from groq import Groq

from config.settings import Settings
from models.rag import ChatTurn, SourceChunk
from services.rag.retriever import retrieve
from utils.errors import ConfigurationError, LLMError
from utils.logging import get_logger

logger = get_logger(__name__)

NO_CONTEXT_MESSAGE = "I couldn't find enough information about that in this meeting."

SYSTEM_PROMPT = """You are a meeting assistant. Answer the user's question using ONLY the \
provided meeting transcript excerpts below. Be concise and direct.

Rules:
- If the excerpts don't contain the answer, say exactly: "{no_context}"
- Never invent names, numbers, or facts not present in the excerpts.
- Use the recent conversation to resolve pronouns/follow-ups (e.g. "it", "that", "they").
- Do not mention "excerpts" or "chunks" in your answer -- just answer naturally, as if you \
attended the meeting.
""".format(no_context=NO_CONTEXT_MESSAGE)


def _build_context(sources: list[SourceChunk]) -> str:
    if not sources:
        return "(no relevant excerpts found)"
    lines = []
    for s in sources:
        label = f"[Chunk {s.chunk_id}" + (f" · {s.timestamp_label()}]" if s.timestamp_label() else "]")
        lines.append(f"{label}\n{s.text}")
    return "\n\n".join(lines)


def _build_messages(
    question: str, sources: list[SourceChunk], history: list[ChatTurn]
) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append(
        {
            "role": "user",
            "content": (
                f"MEETING EXCERPTS:\n{_build_context(sources)}\n\n"
                f"QUESTION: {question}"
            ),
        }
    )
    return messages


class MeetingChat:
    def __init__(self, settings: Settings):
        if not settings.groq_api_key:
            raise ConfigurationError("GROQ_API_KEY is not set.", detail="required for chat")
        self.settings = settings
        self._client = Groq(api_key=settings.groq_api_key)

    def _bounded_history(self, history: list[ChatTurn]) -> list[ChatTurn]:
        n = self.settings.chat_history_turns
        return history[-n:] if n > 0 else []

    def retrieve_sources(self, question: str, meeting_id: str):
        return retrieve(question, meeting_id, self.settings)

    def stream_answer(
        self, question: str, meeting_id: str, history: list[ChatTurn]
    ) -> tuple[Iterator[str], list[SourceChunk], bool, str]:
        """Returns (token_stream, sources, used_fallback, fallback_reason).
        Consume the stream fully to get the complete answer."""

        result = self.retrieve_sources(question, meeting_id)

        if not result.sources:
            def empty_stream():
                yield NO_CONTEXT_MESSAGE
            return empty_stream(), [], result.used_fallback, result.fallback_reason

        messages = _build_messages(question, result.sources, self._bounded_history(history))

        def token_stream() -> Iterator[str]:
            try:
                stream = self._client.chat.completions.create(
                    model=self.settings.groq_chat_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=800,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
            except Exception as exc:  # noqa: BLE001
                logger.error("Chat streaming failed: %s", exc)
                raise LLMError(
                    "The AI couldn't generate an answer right now. Please try again.",
                    detail=str(exc),
                ) from exc

        return token_stream(), result.sources, result.used_fallback, result.fallback_reason
