"""Schema and row validation for the UCI CSV."""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Protocol

import pandas as pd

from .exceptions import SchemaError, ValidationError
from .extract import iter_nonempty_lines

LOGGER = logging.getLogger(__name__)
EXPECTED_TARGETS = {"Dropout", "Enrolled", "Graduate"}
EXPECTED_COLUMNS = 37


class FrameValidator(Protocol):
    """Minimal interface for interchangeable frame-level checks."""

    def validate(self, frame: pd.DataFrame) -> None: ...


@dataclass(frozen=True)
class ValidationFinding:
    """A compact, typed quality observation."""

    rule: str
    count: int
    detail: str


def _width_findings(path: Path, expected: int) -> Iterator[ValidationFinding]:
    for line_number, line in iter_nonempty_lines(path):
        width = len(next(csv.reader([line], delimiter=";")))
        if line_number == 1:
            expected = width
        elif width != expected:
            yield ValidationFinding(
                "row_width", 1, f"line {line_number}: {width} fields, expected {expected}"
            )


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load the original CSV and reject structural or target-value violations."""
    if not path.is_file():
        raise SchemaError(f"Dataset file does not exist: {path}")
    # Stop early after five examples; a malformed file already fails validation.
    findings = list(islice(_width_findings(path, EXPECTED_COLUMNS), 5))
    if findings:
        raise ValidationError(
            "Malformed CSV rows: " + "; ".join(item.detail for item in findings[:5])
        )
    try:
        frame = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    except (UnicodeError, pd.errors.ParserError, OSError) as exc:
        raise SchemaError(f"Could not parse semicolon-delimited UTF-8 CSV: {exc}") from exc
    if len(frame.columns) != EXPECTED_COLUMNS:
        raise SchemaError(f"Expected {EXPECTED_COLUMNS} columns, found {len(frame.columns)}")
    if frame.columns.duplicated().any():
        raise SchemaError("Duplicate column names are not allowed")
    if frame.columns[-1] != "Target":
        raise SchemaError("Final target column must be named 'Target'")
    if frame.columns.str.strip().tolist() != frame.columns.tolist():
        LOGGER.warning(
            "Whitespace is present in one or more original column names; names preserved"
        )
    if frame.isna().any().any():
        missing = frame.isna().sum().loc[lambda values: values > 0].to_dict()
        raise ValidationError(f"Missing values found: {missing}")
    observed = set(frame["Target"].astype(str).unique())
    if observed != EXPECTED_TARGETS:
        raise ValidationError(f"Unexpected target labels: {sorted(observed)}")
    if frame.duplicated().any():
        LOGGER.warning(
            "Found %d duplicate rows; retained pending explicit policy",
            int(frame.duplicated().sum()),
        )
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns[:-1]):
        raise ValidationError(
            "All 36 predictor columns are expected to be numeric codes or measurements"
        )
    LOGGER.info("Validated %d rows and %d columns", len(frame), len(frame.columns))
    return frame
