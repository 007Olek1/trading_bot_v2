"""Custom exception hierarchy for Bybit API interactions."""


class BybitAPIError(Exception):
    """Base exception for Bybit API client errors."""


class BybitRequestError(BybitAPIError):
    """Raised when a request to Bybit fails."""

    def __init__(self, message: str, status_code: int | None = None, payload: dict | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


class BybitValidationError(BybitAPIError):
    """Raised when provided data is invalid for the Bybit API."""


class BybitConfigurationError(BybitAPIError):
    """Raised when client configuration is invalid or incomplete."""

