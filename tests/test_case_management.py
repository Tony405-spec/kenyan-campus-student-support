from datetime import date, timedelta

import pytest

from dropout_pipeline.case_management import (
    CaseStatus,
    SupportCase,
    SupportCategory,
    load_cases,
    overdue_cases,
    upsert_case,
)


def make_case(reference: str = "CASE-DEMO-01", status: str = "Open") -> SupportCase:
    return SupportCase.create(
        reference=reference,
        reviewed_on=date.today(),
        reason="Academic-progress check-in requested",
        category=SupportCategory.ACADEMIC.value,
        next_action="Offer an academic advising appointment",
        responsible_role="Academic adviser",
        follow_up_date=date.today() + timedelta(days=3),
        status=status,
        administrative_note="Demo case; no personal details.",
    )


def test_case_create_persist_and_reload(tmp_path) -> None:
    path = tmp_path / "cases.json"
    created = upsert_case(path, make_case())
    assert len(created) == 1
    assert load_cases(path)[0].reference == "CASE-DEMO-01"
    assert "student ID" not in path.read_text(encoding="utf-8").lower()


def test_upsert_updates_case_and_preserves_creation_time(tmp_path) -> None:
    path = tmp_path / "cases.json"
    initial = upsert_case(path, make_case())[0]
    updated_case = SupportCase.create(
        reference="CASE-DEMO-01",
        reviewed_on=date.today(),
        reason="Follow-up conversation completed",
        category=SupportCategory.ACADEMIC.value,
        next_action="Check progress next week",
        responsible_role="Academic adviser",
        follow_up_date=date.today() + timedelta(days=7),
        status=CaseStatus.IN_PROGRESS.value,
    )
    updated = upsert_case(path, updated_case)[0]
    assert updated.reason == "Follow-up conversation completed"
    assert updated.created_at == initial.created_at
    assert len(load_cases(path)) == 1


def test_overdue_ignores_closed_cases() -> None:
    yesterday = date.today() - timedelta(days=1)
    open_case = SupportCase.create(
        "CASE-OLD-01",
        date.today() - timedelta(days=3),
        "Academic check-in",
        SupportCategory.ACADEMIC.value,
        "Meet adviser",
        "Academic adviser",
        yesterday,
    )
    closed_case = SupportCase.create(
        "CASE-OLD-02",
        date.today() - timedelta(days=3),
        "Academic check-in",
        SupportCategory.ACADEMIC.value,
        "Meet adviser",
        "Academic adviser",
        yesterday,
        status=CaseStatus.CLOSED.value,
    )
    assert [case.reference for case in overdue_cases([open_case, closed_case])] == ["CASE-OLD-01"]


def test_rejects_non_pseudonymous_reference() -> None:
    with pytest.raises(ValueError, match="pseudonymous"):
        SupportCase.create(
            "24003652",
            date.today(),
            "A reason",
            SupportCategory.WELFARE.value,
            "Contact student affairs",
            "Student affairs",
            date.today(),
        )
