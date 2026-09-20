"""
services/analysis/meeting_analyzer.py
===============================================================================
Replaces the old five-call pipeline (clean -> classify -> title -> summarize
-> extract) with ONE structured-JSON call to Groq. Faster, cheaper, and
easier to keep grounded in the transcript.

Flow:

    transcript
        |
        v
    one Groq chat completion, JSON mode, low temperature
        |
        v
    parse as MeetingAnalysis
        |
        +-- parse fails --> bounded repair retry (send the error back)
        |
        +-- still fails  --> MeetingAnalysis.fallback()
===============================================================================
"""

from __future__ import annotations

import json
import time

from groq import Groq
from pydantic import ValidationError as PydanticValidationError

from config.settings import Settings
from models.meeting import MeetingAnalysis
from utils.errors import AnalysisError, ConfigurationError
from utils.logging import get_logger, log_stage

logger = get_logger(__name__)

# Keep the transcript within a safe context budget. ~4 chars/token is a
# reasonable rule of thumb; this keeps prompt + completion comfortably
# inside the model's context window without an extra tokenizer dependency.
MAX_TRANSCRIPT_CHARS = 48_000

SYSTEM_PROMPT = """You are a precise meeting-intelligence analyst.
Read the transcript and return ONLY a single JSON object matching this exact schema \
(no markdown fences, no commentary, no extra keys):

{
  "title": string,
  "content_type": string,          // e.g. "meeting", "lecture", "interview", "standup"
  "language": string,               // best-effort language name of the transcript
  "executive_summary": string,      // 3-6 sentences
  "key_points": [string],
  "action_items": [
    {"task": string, "owner": string|null, "deadline": string|null, "priority": string|null, "status": string|null}
  ],
  "decisions": [string],
  "open_questions": [string],
  "risks_and_blockers": [string],
  "topics": [string],
  "next_steps": [string],
  "sentiment": string,              // "positive" | "neutral" | "negative" | "mixed"
  "meeting_outcome": string         // one sentence
}

Rules:
- Every item you extract must be directly supported by the transcript. Do not invent names, \
dates, or facts that are not present.
- If a section has nothing relevant, return an empty list (or empty string), never a placeholder.
- Keep "executive_summary" and "meeting_outcome" concise and factual.
- Output valid JSON only.
"""


def _build_user_prompt(transcript: str) -> str:
    truncated = transcript.strip()
    note = ""
    if len(truncated) > MAX_TRANSCRIPT_CHARS:
        truncated = truncated[:MAX_TRANSCRIPT_CHARS]
        note = "\n\n[Note: transcript truncated for length. Analyze what's provided.]"
    return f"TRANSCRIPT:\n{truncated}{note}"


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")
    return json.loads(raw[start : end + 1])


class MeetingAnalyzer:
    def __init__(self, settings: Settings, *, high_quality: bool = False):
        if not settings.groq_api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is not set.", detail="required for meeting analysis"
            )
        self.model = (
            settings.groq_chat_model_high_quality if high_quality else settings.groq_chat_model
        )
        self._client = Groq(api_key=settings.groq_api_key)

    def _call(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=3000,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    def analyze(self, transcript: str) -> MeetingAnalysis:
        if not transcript or not transcript.strip():
            raise AnalysisError("Transcript is empty; nothing to analyze.")

        started = time.perf_counter()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(transcript)},
        ]
        raw = ""

        try:
            raw = self._call(messages)
            data = _extract_json(raw)
            analysis = MeetingAnalysis.model_validate(data)
            log_stage(
                logger, "analysis", duration=time.perf_counter() - started,
                model=self.model, repaired=False,
            )
            return analysis

        except (json.JSONDecodeError, ValueError, PydanticValidationError) as first_error:
            logger.warning("Analysis JSON parse/validation failed, attempting repair: %s", first_error)

            try:
                repair_messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "That response was not valid JSON matching the schema. "
                            f"Error: {first_error}. Return ONLY the corrected JSON object."
                        ),
                    },
                ]
                raw_repair = self._call(repair_messages)
                data = _extract_json(raw_repair)
                analysis = MeetingAnalysis.model_validate(data)
                log_stage(
                    logger, "analysis", duration=time.perf_counter() - started,
                    model=self.model, repaired=True,
                )
                return analysis

            except Exception as second_error:  # noqa: BLE001
                log_stage(
                    logger, "analysis", duration=time.perf_counter() - started,
                    status="fallback", model=self.model,
                )
                logger.error("Analysis repair also failed, using fallback schema: %s", second_error)
                return MeetingAnalysis.fallback(transcript)

        except Exception as exc:  # noqa: BLE001
            log_stage(
                logger, "analysis", duration=time.perf_counter() - started,
                status="failed", model=self.model,
            )
            raise AnalysisError(
                "Meeting analysis failed. The AI service may be temporarily unavailable.",
                detail=str(exc),
            ) from exc
