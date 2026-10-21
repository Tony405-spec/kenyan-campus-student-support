"""Small local JSON register for pseudonymous student-support demonstrations."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class CaseStatus(StrEnum):
    """Human-owned follow-up states."""

    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    REFERRED = "Referred"
    CLOSED = "Closed"


class SupportCategory(StrEnum):
    """Non-clinical administrative follow-up categories."""

    ACADEMIC = "Academic progress"
    FINANCIAL = "Fees or financial aid"
    ATTENDANCE = "Attendance or engagement"
    ACADEMIC_SUPPORT = "Access to academic support"
    WELFARE = "General student welfare"


@dataclass(frozen=True, slots=True)
class SupportCase:
    """A minimal pseudonymous record; never store credentials or sensitive details."""

    reference: str
    reviewed_on: str
    reason: str
    category: str
    next_action: str
    responsible_role: str
    follow_up_date: str
    status: str = CaseStatus.OPEN.value
    administrative_note: str = ""
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"CASE-[A-Z0-9-]{3,24}", self.reference):
            raise ValueError("Use a pseudonymous reference such as CASE-DEMO-01")
        for field, value, limit in (
            ("Reason", self.reason, 300),
            ("Next action", self.next_action, 300),
            ("Responsible role", self.responsible_role, 100),
            ("Administrative note", self.administrative_note, 500),
        ):
            if field != "Administrative note" and not value.strip():
                raise ValueError(f"{field} is required")
            if len(value) > limit:
                raise ValueError(f"{field} must be at most {limit} characters")
        date.fromisoformat(self.reviewed_on)
        date.fromisoformat(self.follow_up_date)
        if self.category not in {item.value for item in SupportCategory}:
            raise ValueError("Unsupported student-support category")
        if self.status not in {item.value for item in CaseStatus}:
            raise ValueError("Unsupported case status")

    @classmethod
    def create(
        cls,
        reference: str,
        reviewed_on: date,
        reason: str,
        category: str,
        next_action: str,
        responsible_role: str,
        follow_up_date: date,
        status: str = CaseStatus.OPEN.value,
        administrative_note: str = "",
    ) -> SupportCase:
        """Create an auditable record with UTC timestamps."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        return cls(
            reference=reference.strip().upper(),
            reviewed_on=reviewed_on.isoformat(),
            reason=reason.strip(),
            category=category,
            next_action=next_action.strip(),
            responsible_role=responsible_role.strip(),
            follow_up_date=follow_up_date.isoformat(),
            status=status,
            administrative_note=administrative_note.strip(),
            created_at=now,
            updated_at=now,
        )


def load_cases(path: Path) -> list[SupportCase]:
    """Load and validate local cases; an invalid file raises instead of disappearing."""
    if not path.exists():
        return []
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Case file must contain a JSON list")
        return [SupportCase(**record) for record in raw]
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        raise ValueError(f"Could not load support-case file {path}: {exc}") from exc


def save_cases(path: Path, cases: list[SupportCase]) -> None:
    """Atomically persist validated cases to the configured local demonstration file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps([asdict(case) for case in cases], indent=2, ensure_ascii=False)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def upsert_case(path: Path, case: SupportCase) -> list[SupportCase]:
    """Create or replace a case by pseudonymous reference and persist the result."""
    cases = load_cases(path)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    updated = replace(case, updated_at=now)
    for index, existing in enumerate(cases):
        if existing.reference == case.reference:
            updated = replace(updated, created_at=existing.created_at)
            cases[index] = updated
            break
    else:
        cases.append(updated)
    save_cases(path, cases)
    return cases


def overdue_cases(cases: list[SupportCase], today: date | None = None) -> list[SupportCase]:
    """Return non-closed cases whose scheduled follow-up date has passed."""
    current_day = today or date.today()
    return [
        case
        for case in cases
        if case.status != CaseStatus.CLOSED.value
        and date.fromisoformat(case.follow_up_date) < current_day
    ]
