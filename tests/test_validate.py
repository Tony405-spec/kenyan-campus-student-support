"""Schema and row validation tests."""

from pathlib import Path

import pytest

from dropout_pipeline.exceptions import SchemaError, ValidationError
from dropout_pipeline.validate import load_and_validate


def test_load_valid_csv(tmp_path: Path, sample_frame) -> None:
    path = tmp_path / "data.csv"
    sample_frame.to_csv(path, sep=";", index=False)
    loaded = load_and_validate(path)
    assert loaded.shape == (3, 37)


@pytest.mark.parametrize(
    "contents, error",
    [
        ("a;b\n1;2\n", SchemaError),
        ("a;b;Target\n1;2;Unknown\n", SchemaError),
    ],
)
def test_invalid_schema(tmp_path: Path, contents: str, error: type[Exception]) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(error):
        load_and_validate(path)


def test_rejects_missing_cells(tmp_path: Path, sample_frame) -> None:
    sample_frame.loc[0, "GDP"] = float("nan")
    path = tmp_path / "missing.csv"
    sample_frame.to_csv(path, sep=";", index=False)
    with pytest.raises(ValidationError, match="Missing values"):
        load_and_validate(path)


def test_rejects_malformed_row_width(tmp_path: Path, sample_frame) -> None:
    path = tmp_path / "malformed.csv"
    sample_frame.to_csv(path, sep=";", index=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("only;two;fields\n")
    with pytest.raises(ValidationError, match="Malformed CSV rows"):
        load_and_validate(path)
