"""Tests for localized intent routing and crisis signposting."""

import pytest

from dropout_pipeline.support_chatbot import SupportIntent, classify_intent, reply_to


@pytest.mark.parametrize(
    "message, expected",
    [
        ("I have two CATs next week and haven't started revision", SupportIntent.ACADEMIC),
        ("HELB has not come through and I'm worried about fees", SupportIntent.FINANCIAL),
        ("Niko solo in the hostel", SupportIntent.LONELINESS),
        ("Nina wasiwasi about exams", SupportIntent.ANXIETY),
        ("I feel hopeless and empty", SupportIntent.DEPRESSION),
        ("Habari, can you help?", SupportIntent.GREETING),
        ("Something else is happening", SupportIntent.GENERAL),
    ],
)
def test_localized_intent_examples(message: str, expected: SupportIntent) -> None:
    assert classify_intent(message) == expected


def test_crisis_routing_has_emergency_resources() -> None:
    response = reply_to("I want to hurt myself tonight")
    assert response.intent == SupportIntent.CRISIS
    assert "112 or 999" in response.message
    assert "emergency" in response.resources


def test_financial_reply_warns_against_sharing_credentials() -> None:
    response = reply_to("I have a HELB fee balance and no upkeep")
    assert response.intent == SupportIntent.FINANCIAL
    assert "M-Pesa PIN" in response.message
    assert "helb" in response.resources
