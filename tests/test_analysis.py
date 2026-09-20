"""Tests for services/analysis/meeting_analyzer.py -- JSON parsing, schema
validation, and the fallback path, using a mocked Groq client so no network
calls are made."""

from unittest.mock import MagicMock, patch

import pytest

from models.meeting import MeetingAnalysis
from services.analysis.meeting_analyzer import MeetingAnalyzer, _extract_json
from utils.errors import AnalysisError


VALID_JSON = """{
  "title": "Sprint Planning",
  "content_type": "meeting",
  "language": "en",
  "executive_summary": "The team planned the next sprint.",
  "key_points": ["Reviewed backlog"],
  "action_items": [{"task": "Write tests", "owner": "Alex", "deadline": "Friday", "priority": "high", "status": "pending"}],
  "decisions": ["Adopt trunk-based development"],
  "open_questions": [],
  "risks_and_blockers": [],
  "topics": ["Sprint Planning"],
  "next_steps": ["Start sprint Monday"],
  "sentiment": "positive",
  "meeting_outcome": "Sprint planned successfully."
}"""


def _mock_response(content: str):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def _make_analyzer():
    with patch("services.analysis.meeting_analyzer.Groq"):
        settings = MagicMock(groq_api_key="test-key", groq_chat_model="test-model",
                              groq_chat_model_high_quality="test-model-hq")
        analyzer = MeetingAnalyzer(settings)
    return analyzer


def test_extract_json_handles_markdown_fences():
    raw = "```json\n{\"a\": 1}\n```"
    assert _extract_json(raw) == {"a": 1}


def test_extract_json_raises_without_braces():
    with pytest.raises(ValueError):
        _extract_json("no json here")


def test_analyze_valid_response():
    analyzer = _make_analyzer()
    analyzer._client.chat.completions.create.return_value = _mock_response(VALID_JSON)

    result = analyzer.analyze("Some transcript text about sprint planning.")

    assert isinstance(result, MeetingAnalysis)
    assert result.title == "Sprint Planning"
    assert result.action_items[0].owner == "Alex"


def test_analyze_malformed_json_triggers_repair_then_succeeds():
    analyzer = _make_analyzer()
    analyzer._client.chat.completions.create.side_effect = [
        _mock_response("not valid json"),
        _mock_response(VALID_JSON),
    ]

    result = analyzer.analyze("Some transcript.")
    assert result.title == "Sprint Planning"
    assert analyzer._client.chat.completions.create.call_count == 2


def test_analyze_falls_back_when_repair_also_fails():
    analyzer = _make_analyzer()
    analyzer._client.chat.completions.create.side_effect = [
        _mock_response("still not json"),
        _mock_response("still not json either"),
    ]

    result = analyzer.analyze("Some transcript.")
    assert isinstance(result, MeetingAnalysis)
    assert result.title == "Untitled Meeting"


def test_analyze_empty_transcript_raises():
    analyzer = _make_analyzer()
    with pytest.raises(AnalysisError):
        analyzer.analyze("   ")


def test_analyze_missing_fields_still_validates_with_defaults():
    analyzer = _make_analyzer()
    minimal_json = '{"title": "Quick Sync"}'
    analyzer._client.chat.completions.create.return_value = _mock_response(minimal_json)

    result = analyzer.analyze("A short transcript.")
    assert result.title == "Quick Sync"
    assert result.action_items == []
