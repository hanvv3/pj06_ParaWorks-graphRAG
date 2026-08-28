import logging
import os
import re
from dataclasses import dataclass

from langchain_core.globals import get_debug

CREDENTIAL_SCAN_VERSION = 'credential-scan:v1'

_HIGH_CONFIDENCE_PATTERNS = (
    re.compile(
        r'\b(?:OPENAI_API_KEY|GOOGLE_API_KEY|GEMINI_API_KEY|SLACK_TOKEN)\s*=\s*\S+',
        re.I,
    ),
    re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{20,}\b'),
)


@dataclass(frozen=True)
class InputSafetyDecision:
    allowed: bool
    reason_code: str | None
    scanner_version: str = CREDENTIAL_SCAN_VERSION


def scan_auto_review_plaintext(value: str) -> InputSafetyDecision:
    for pattern in _HIGH_CONFIDENCE_PATTERNS:
        if pattern.search(value):
            return InputSafetyDecision(False, 'sensitive_input_detected')
    return InputSafetyDecision(True, None)


def provider_logging_is_safe() -> bool:
    return (
        get_debug() is False
        and os.getenv('OPENAI_LOG', '').strip().casefold() != 'debug'
        and logging.getLogger('openai').getEffectiveLevel() > logging.DEBUG
    )
