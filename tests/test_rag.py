"""Tests for RAG chunking, the in-memory fallback store (meeting isolation),
and lexical retrieval -- all network-free."""

from models.rag import TranscriptChunk
from models.transcript import Transcript, TranscriptSegment
from services.rag.chunking import chunk_transcript
from services.rag.memory_store import ChunkMemoryStore
from services.rag.retriever import lexical_search


def _transcript_with_segments():
    segments = [
        TranscriptSegment(start=0, end=10, text="We decided to launch the new API on Friday."),
        TranscriptSegment(start=10, end=20, text="Rahul will own the deployment and rollback plan."),
        TranscriptSegment(start=20, end=30, text="The onboarding drop-off rate is a risk we must track."),
    ]
    return Transcript(text=" ".join(s.text for s in segments), language="en", segments=segments)


def test_chunk_transcript_preserves_timestamps():
    transcript = _transcript_with_segments()
    chunks = chunk_transcript(transcript, "meeting-1", chunk_size_words=5, overlap_words=0)
    assert len(chunks) >= 1
    assert all(c.meeting_id == "meeting-1" for c in chunks)
    assert chunks[0].start == 0


def test_chunk_transcript_falls_back_to_text_when_no_segments():
    transcript = Transcript(text="one two three four five six seven eight nine ten")
    chunks = chunk_transcript(transcript, "meeting-2", chunk_size_words=4, overlap_words=1)
    assert len(chunks) > 1
    assert chunks[0].start is None


def test_chunk_transcript_empty_text_returns_no_chunks():
    transcript = Transcript(text="")
    chunks = chunk_transcript(transcript, "meeting-3")
    assert chunks == []


def test_memory_store_meeting_isolation():
    store = ChunkMemoryStore()
    store.put("meeting-a", [TranscriptChunk(meeting_id="meeting-a", chunk_id=0, text="alpha content")])
    store.put("meeting-b", [TranscriptChunk(meeting_id="meeting-b", chunk_id=0, text="beta content")])

    a_chunks = store.get("meeting-a")
    b_chunks = store.get("meeting-b")

    assert len(a_chunks) == 1 and a_chunks[0].text == "alpha content"
    assert len(b_chunks) == 1 and b_chunks[0].text == "beta content"
    assert store.get("meeting-unknown") == []


def test_lexical_search_finds_matching_chunk():
    chunks = [
        TranscriptChunk(meeting_id="m", chunk_id=0, text="The deployment is scheduled for Friday."),
        TranscriptChunk(meeting_id="m", chunk_id=1, text="Marketing discussed the new logo design."),
    ]
    results = lexical_search("When is the deployment happening?", chunks, top_k=5)
    assert results
    assert results[0].chunk_id == 0


def test_lexical_search_no_relevant_result_returns_empty():
    chunks = [
        TranscriptChunk(meeting_id="m", chunk_id=0, text="We discussed quarterly budget numbers."),
    ]
    results = lexical_search("xyz completely unrelated zzzz", chunks, top_k=5)
    assert results == []
