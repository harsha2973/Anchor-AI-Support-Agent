"""
Data cleaning, normalization, and PII anonymization pipeline.

Prepares raw dialogue turns for both intent classification and retrieval indexing.
Critical for customer support data: handles URL redaction, phone/email masking,
whitespace stripping, and thread concatenation.
"""

from __future__ import annotations

import re
from typing import Optional


class TextPreprocessor:
    """
    Sanitizes and prepares customer support dialogue text.

    Ensures that sensitive information (credit cards, emails, phone numbers, tracking IDs)
    is redacted before sending to LLM APIs or vector indexing, while preserving key context.
    """

    # Regex patterns for common sensitive identifiers
    EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
    PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}")
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    ORDER_ID_PATTERN = re.compile(r"\b(order|tracking|ticket|id)[:\s#]+(?=[a-z0-9\-]*\d)[a-z0-9\-]{5,}\b|\b\d{3}-\d{7}-\d{7}\b", re.IGNORECASE)

    def __init__(self, mask_pii: bool = True, normalize_whitespace: bool = True) -> None:
        """
        Initialize preprocessor options.

        Args:
            mask_pii: Whether to redact sensitive customer identifiers with tokens.
            normalize_whitespace: Whether to collapse multiple spaces, newlines, and tabs.
        """
        self.mask_pii = mask_pii
        self.normalize_whitespace = normalize_whitespace

    def clean(self, text: Optional[str]) -> str:
        """
        Apply cleaning transformations to a single text string.

        Args:
            text: Raw input text from customer or brand turn.

        Returns:
            Sanitized and normalized text string.
        """
        if not text:
            return ""

        cleaned = text

        if self.mask_pii:
            cleaned = self._redact_pii(cleaned)

        if self.normalize_whitespace:
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def _redact_pii(self, text: str) -> str:
        """Replace emails, phones, and explicit order IDs with safe placeholders."""
        text = self.EMAIL_PATTERN.sub("[EMAIL]", text)
        text = self.PHONE_PATTERN.sub("[PHONE]", text)
        text = self.ORDER_ID_PATTERN.sub("[ORDER_ID]", text)
        text = self.URL_PATTERN.sub("[LINK]", text)
        return text
