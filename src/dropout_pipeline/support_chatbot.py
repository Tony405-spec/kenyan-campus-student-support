"""Local, rule-based wellbeing support routing for a Kenyan campus demo.

This module offers supportive signposting only. It does not diagnose, store,
or replace a qualified counsellor, clinician, emergency service, or campus team.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class SupportIntent(StrEnum):
    """Intent categories explicitly supported by the prototype."""

    CRISIS = "immediate_safety"
    DEPRESSION = "low_mood_or_depression"
    FINANCIAL = "financial_pressure"
    ANXIETY = "anxiety_or_worry"
    ACADEMIC = "academic_or_exam_pressure"
    LONELINESS = "loneliness_or_isolation"
    GREETING = "greeting"
    THANKS = "thanks"
    GENERAL = "general_support"


@dataclass(frozen=True, slots=True)
class SupportReply:
    """A deterministic reply, intent label, and safe next-step identifiers."""

    intent: SupportIntent
    message: str
    resources: tuple[str, ...] = ()


_PATTERNS: tuple[tuple[SupportIntent, re.Pattern[str]], ...] = (
    (
        SupportIntent.CRISIS,
        re.compile(
            r"\b(suicid\w*|kill myself|end my life|want to die|self[ -]?harm|hurt myself|"
            r"harm myself|can't stay safe|cannot stay safe|kujiumiza|kujiua|sitaki kuishi)\b",
        ),
    ),
    (
        SupportIntent.FINANCIAL,
        re.compile(
            r"\b(helb|fees?|fee balance|tuition|rent|hostel fee|no fare|sina nauli|"
            r"can't afford|cannot afford|financial|money stress|upkeep|bursary|"
            r"school fees|fee clearance)\b",
        ),
    ),
    (
        SupportIntent.DEPRESSION,
        re.compile(
            r"\b(depress\w*|hopeless|no point|nothing matters|empty|low mood|"
            r"can't get out of bed|cannot get out of bed|huzuni|sina matumaini)\b",
        ),
    ),
    (
        SupportIntent.ANXIETY,
        re.compile(
            r"\b(anxious|anxiety|panic|panicking|worried|worrying|overthink\w*|"
            r"heart racing|can't sleep|cannot sleep|waswasi|nina wasiwasi|nervous)\b",
        ),
    ),
    (
        SupportIntent.ACADEMIC,
        re.compile(
            r"\b(exam\w*|cat\b|test\w*|revision|backlog|deadline|assignment|"
            r"coursework|resit|retake|academic pressure|too much coursework|"
            r"study pressure|sijasoma|nimelemewa na masomo)\b",
        ),
    ),
    (
        SupportIntent.LONELINESS,
        re.compile(
            r"\b(lonely|loneliness|alone|isolated|no friends|niko solo|"
            r"miss home|homesick|sina marafiki|mpweke)\b",
        ),
    ),
    (SupportIntent.GREETING, re.compile(r"\b(hi|hello|hey|habari|niaje)\b")),
    (SupportIntent.THANKS, re.compile(r"\b(thanks|thank you|asante|shukran)\b")),
)


_RESPONSES: dict[SupportIntent, SupportReply] = {
    SupportIntent.CRISIS: SupportReply(
        SupportIntent.CRISIS,
        "I’m really sorry this feels so heavy. If you might act on thoughts of suicide or "
        "self-harm, or you cannot stay safe right now, call Kenya emergency services on "
        "112 or 999, go to the nearest health facility, and tell someone you trust who can "
        "stay with you. You do not have to handle this alone. This chatbot cannot provide "
        "emergency care.",
        ("emergency", "lvct", "red_cross_chat"),
    ),
    SupportIntent.DEPRESSION: SupportReply(
        SupportIntent.DEPRESSION,
        "That sounds painful, and you deserve support. I can’t diagnose depression, but "
        "feeling low or hopeless is worth sharing with a counsellor, health professional, "
        "or trusted person. Could you contact your campus student-welfare/student-affairs "
        "office today and ask for the current counselling referral? If you may harm yourself "
        "or cannot stay safe, use the immediate-safety options on the Resources page.",
        ("campus_support", "lvct", "red_cross_chat", "emergency"),
    ),
    SupportIntent.FINANCIAL: SupportReply(
        SupportIntent.FINANCIAL,
        "Fee, rent, food and transport pressure can make campus life feel unmanageable. "
        "For a practical next step, contact your campus student-finance office, student "
        "affairs office, or department and ask about payment plans, bursary referrals, "
        "fee-clearance options, or a short-term support pathway. Check HELB/HEF status only "
        "through the official portal. Please don’t share your ID, HELB login, M-Pesa PIN, "
        "or bank details in this chatbot.",
        ("campus_support", "helb"),
    ),
    SupportIntent.ANXIETY: SupportReply(
        SupportIntent.ANXIETY,
        "Worry can feel intense, especially with exams, fees or family expectations in the "
        "mix. If it feels okay, place both feet on the floor, breathe out slowly, and name "
        "five things you can see and four things you can feel. Then choose one small next "
        "step. If this keeps disrupting sleep or study, ask a campus support office or "
        "qualified health professional for help.",
        ("campus_support", "lvct", "red_cross_chat"),
    ),
    SupportIntent.ACADEMIC: SupportReply(
        SupportIntent.ACADEMIC,
        "A pile-up of CATs, assignments or revision can be overwhelming. Pick one unit and "
        "one topic; work on it for 25 minutes, take a short break, then decide the next "
        "step. A classmate or study group can help you stay accountable. If deadlines or "
        "missed classes are becoming unmanageable, contact your lecturer, academic adviser "
        "or department early and ask what options are available.",
        ("campus_support",),
    ),
    SupportIntent.LONELINESS: SupportReply(
        SupportIntent.LONELINESS,
        "Feeling alone in a hostel or away from home can be hard, and it doesn’t mean you’ve "
        "failed to fit in. Could you send one classmate a simple message, join a study group, "
        "student society, sports activity or faith/community group that feels comfortable? "
        "If isolation is affecting your wellbeing, a campus support office can help you find "
        "a confidential human referral.",
        ("campus_support", "lvct"),
    ),
    SupportIntent.GREETING: SupportReply(
        SupportIntent.GREETING,
        "Hi, karibu. I can help you find a small next step for exam pressure, anxiety, "
        "financial stress, loneliness or low mood. What has been hardest on campus lately?",
    ),
    SupportIntent.THANKS: SupportReply(
        SupportIntent.THANKS,
        "You’re welcome. You deserve support from people around you too. You can keep this "
        "chat general, or open the Resources page for verified Kenyan contacts.",
        ("campus_support", "lvct", "red_cross_chat"),
    ),
    SupportIntent.GENERAL: SupportReply(
        SupportIntent.GENERAL,
        "Thanks for sharing that. I’m a campus support signpost, not a counsellor or "
        "diagnostic tool. You could tell me whether this is mainly about studies, money, "
        "worry, feeling alone or low mood; I’ll suggest a practical next step and human "
        "support option.",
        ("campus_support", "lvct", "red_cross_chat"),
    ),
}


def classify_intent(message: str) -> SupportIntent:
    """Return the first matching intent; immediate-safety patterns are checked first."""
    normalized = " ".join(message.lower().split())
    for intent, pattern in _PATTERNS:
        if pattern.search(normalized):
            return intent
    return SupportIntent.GENERAL


def reply_to(message: str) -> SupportReply:
    """Return a fixed, safety-aware reply without sending text to a remote service."""
    return _RESPONSES[classify_intent(message)]
