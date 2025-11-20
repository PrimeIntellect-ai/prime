"""Exceptions for Prime Evals SDK."""


class EvalsAPIError(Exception):
    """Base exception for Evals API errors."""

    pass


class EnvironmentNotFoundError(EvalsAPIError):
    """Raised when an environment is not found in the hub."""

    pass


class EvaluationNotFoundError(EvalsAPIError):
    """Raised when an evaluation is not found."""

    pass


class InvalidSampleError(EvalsAPIError):
    """Raised when a sample has invalid data."""

    pass


class InvalidEvaluationError(EvalsAPIError):
    """Raised when evaluation configuration is invalid."""

    pass
