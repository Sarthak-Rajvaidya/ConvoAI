"""
utils/demo_data.py
===============================================================================
Sample meeting used by Demo Mode (DEMO_MODE=true) so the product can be
shown end-to-end -- analysis, RAG, chat, reports -- without consuming API
credits or needing a real recording. Useful for portfolio/interview demos.
===============================================================================
"""

from __future__ import annotations

import uuid

from models.meeting import ActionItem, MeetingAnalysis, MeetingResult
from models.transcript import Transcript, TranscriptSegment

DEMO_TRANSCRIPT_SEGMENTS = [
    (0, 18, "Alright, let's get started. Today we're reviewing the Q3 roadmap and the "
            "deployment plan for the new payments API."),
    (18, 42, "Priya, can you walk us through where the payments API integration stands? "
             "Last week we said staging tests would wrap up by Wednesday."),
    (42, 71, "Staging tests are done as of yesterday. We found one issue with retry "
             "handling on failed transactions, but that's fixed now. I think we're good "
             "to deploy to production by Friday."),
    (71, 96, "Great. Let's make Friday the deployment deadline then. Rahul, can you own "
             "the production deployment and coordinate the rollback plan just in case?"),
    (96, 110, "Yeah, I can take that. I'll have the rollback plan documented by Thursday "
              "and share it in the channel."),
    (110, 138, "One open question -- we still haven't decided whether we're doing a "
               "canary rollout or a full deploy on Friday. I'd lean canary given this "
               "touches payments."),
    (138, 160, "Agreed, let's do a canary rollout, 10 percent of traffic first, then "
               "ramp up over the weekend if metrics look clean."),
    (160, 185, "Okay, decision made: canary rollout starting Friday, full rollout Monday "
               "if error rates stay under threshold."),
    (185, 210, "Switching topics -- the mobile team flagged that the new onboarding flow "
               "is seeing a higher drop-off than expected, around 22 percent at the "
               "verification step."),
    (210, 235, "That's a risk we should track. Can someone look into whether it's the SMS "
               "verification provider or a UI issue? I don't think we have an owner for "
               "that yet."),
    (235, 250, "I'll take a first pass at the analytics this week and report back."),
    (250, 275, "Sounds good. For next steps: Priya finalizes deployment by Friday, Rahul "
               "owns rollback plan and the canary rollout, and we revisit onboarding "
               "drop-off numbers next week."),
    (275, 300, "One more thing -- overall the team is feeling good about the Q3 progress. "
               "We're basically on track except for this onboarding question. Let's wrap "
               "up here, thanks everyone."),
]


def build_demo_transcript() -> Transcript:
    segments = [
        TranscriptSegment(start=float(s), end=float(e), text=t)
        for s, e, t in DEMO_TRANSCRIPT_SEGMENTS
    ]
    text = " ".join(seg.text for seg in segments)
    return Transcript(
        text=text, language="en", segments=segments,
        provider="demo", model="demo", duration_seconds=300.0,
    )


def build_demo_analysis() -> MeetingAnalysis:
    return MeetingAnalysis(
        title="Q3 Roadmap & Payments API Deployment",
        content_type="meeting",
        language="en",
        executive_summary=(
            "The team confirmed the payments API is ready for production after staging "
            "tests passed, and agreed to a canary rollout starting Friday with a full "
            "rollout on Monday if metrics stay clean. A separate risk was raised: the new "
            "onboarding flow has a higher-than-expected drop-off at the verification "
            "step, which needs investigation. Overall Q3 progress is on track."
        ),
        key_points=[
            "Staging tests for the payments API integration are complete; one retry-handling bug was found and fixed.",
            "Deployment to production is planned for Friday using a canary rollout.",
            "Mobile onboarding flow is seeing ~22% drop-off at the SMS verification step.",
        ],
        action_items=[
            ActionItem(task="Finalize and deploy the payments API to production", owner="Priya", deadline="Friday", priority="high", status="in progress"),
            ActionItem(task="Document the rollback plan and share in the team channel", owner="Rahul", deadline="Thursday", priority="high", status="in progress"),
            ActionItem(task="Own the canary rollout and monitor error rates over the weekend", owner="Rahul", deadline="Monday", priority="high", status="pending"),
            ActionItem(task="Analyze onboarding drop-off to determine if it's the SMS provider or a UI issue", owner="Unassigned (volunteer to follow up)", deadline="This week", priority="medium", status="pending"),
        ],
        decisions=[
            "Deploy the payments API using a canary rollout: 10% of traffic on Friday, ramping up over the weekend.",
            "Full rollout on Monday if error rates stay under threshold.",
        ],
        open_questions=[
            "Is the onboarding drop-off caused by the SMS verification provider or a UI issue?",
        ],
        risks_and_blockers=[
            "Higher-than-expected drop-off (~22%) at the onboarding verification step.",
        ],
        topics=["Payments API", "Deployment", "Canary Rollout", "Onboarding", "Q3 Roadmap"],
        next_steps=[
            "Priya finalizes and deploys the payments API by Friday.",
            "Rahul documents the rollback plan and owns the canary rollout.",
            "Revisit onboarding drop-off numbers next week.",
        ],
        sentiment="positive",
        meeting_outcome=(
            "Team aligned on a Friday canary deployment for the payments API and flagged "
            "onboarding drop-off as a risk to investigate."
        ),
    )


def build_demo_result() -> MeetingResult:
    return MeetingResult(
        meeting_id=f"demo-{uuid.uuid4().hex[:8]}",
        source_label="Demo Meeting — Q3 Roadmap Review",
        analysis=build_demo_analysis(),
        transcript_text=build_demo_transcript().text,
        transcript_language="en",
        transcript_provider="demo",
        duration_seconds=300.0,
        timings={"audio_acquisition": 0.0, "transcription": 0.0, "analysis": 0.0, "total": 0.0},
    )
