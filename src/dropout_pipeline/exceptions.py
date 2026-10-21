"""Small, actionable exception hierarchy for pipeline failures."""


class PipelineError(Exception):
    """Base class for expected pipeline errors."""


class DownloadError(PipelineError):
    """The source archive could not be downloaded after bounded retries."""


class InvalidArchiveError(PipelineError):
    """The downloaded file is not a usable UCI ZIP archive."""


class SchemaError(PipelineError):
    """The input data does not match the expected dataset structure."""


class ValidationError(PipelineError):
    """One or more row-level data quality rules failed."""
