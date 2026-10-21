"""Extraction and utility behaviour."""

from pathlib import Path

from dropout_pipeline.extract import bounded_retry, iter_nonempty_lines


def test_generator_skips_empty_lines(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("one\n\n two\n", encoding="utf-8")
    assert list(iter_nonempty_lines(path)) == [(1, "one"), (3, " two")]


def test_retry_is_bounded_and_preserves_function_metadata() -> None:
    calls = 0

    @bounded_retry(attempts=2, delay=0)
    def flaky() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary")
        return 7

    assert flaky.__name__ == "flaky"
    assert flaky() == 7
    assert calls == 2
