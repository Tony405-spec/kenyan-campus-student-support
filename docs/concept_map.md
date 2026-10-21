# Advanced Python concept map

| Concept | Where and purpose | Test / simpler alternative |
|---|---|---|
| Context managers | `extract.py`: `with` closes URL response, ZIP and CSV streams even on errors. `sha256_file` streams safely. | Extraction tests; simpler manual `open`/`close` needs `try/finally`. |
| Type hints | Public functions, data structures, protocol and CLI use concrete `Path`, `DataFrame`, `Series`, arrays and typed results. | `mypy src/`; simpler untyped functions defer mistakes until runtime. |
| Protocol | `FrameValidator` and `Predictor` define small structural contracts for swapping implementations. | Existing validators/estimators satisfy methods; simpler direct calls bind code more tightly. |
| Dataclasses | Frozen `PipelineConfig`, `DatasetParts`, `ModelResult`, `ValidationFinding` name related values and prevent accidental config mutation. | Used in actual pipeline; tuple/dict alternatives are less self-documenting. |
| Enum | `RunStage` gives a bounded vocabulary for pipeline milestones. | Static reading; plain strings are simpler but allow typos. |
| Generator/iterator | `iter_nonempty_lines` validates row widths lazily, one line at a time. | `test_generator_skips_empty_lines`; reading all lines uses more memory. Pandas vectorisation is better for numeric whole-column transforms. |
| Decorators | `bounded_retry` retries transient download failures a fixed number of times and uses `functools.wraps`. | `test_retry_is_bounded_and_preserves_function_metadata`; inline retry logic is simpler for one call but duplicates policy. |
| Pattern matching | `_interpret_outcome` in `evaluate.py` routes three known target labels for figure display. | `test_outcome_routes`; `if/elif` is a straightforward alternative. |
| Logging | Modules log progress and quality observations without printing inside reusable code. | CLI invocation shows log output; `print` is less configurable for a library. |
| Custom exceptions | `PipelineError` branches distinguish download, archive, schema and validation problems. | Validation tests; built-in exceptions alone offer less domain context. |
| Functional utilities | `functools.wraps`, `ParamSpec` and `itertools.islice` preserve a reusable retry wrapper's metadata and cap malformed-row diagnostics at five examples. | Retry test, generator test and malformed-row test; a manual counter is a simpler alternative. |
| Automated testing | Pytest fixtures, parameterized invalid schemas, success/failure cases across modules. | `python -m pytest`; manual checks are not repeatable. |

## DataFrame trade-off

The dataset is rectangular numerical/categorical tabular data. A pandas DataFrame efficiently expresses columns and works directly with sklearn. Dataclasses represent configuration and metadata, not individual student rows; creating thousands of Python objects would add overhead without analytical benefit.

