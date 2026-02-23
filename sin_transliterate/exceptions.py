# sin_transliterate/exceptions.py


class SinTransliterateError(Exception):
    """Base exception for all sin_transliterate errors."""


class InvalidModelError(SinTransliterateError):
    """Raised when an unrecognised model type is specified."""


class ModelLoadError(SinTransliterateError):
    """Raised when the model cannot be fetched or instantiated."""


class TransliterationError(SinTransliterateError):
    """Raised when inference fails on the provided input."""
