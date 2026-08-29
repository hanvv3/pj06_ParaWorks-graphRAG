import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass

from langchain_core.globals import get_debug

CREDENTIAL_SCAN_VERSION = 'credential-scan:v1'
HIGH_RISK_LANGUAGE_VERSION = 'auto-review-high-risk-language:v1'

_HIGH_CONFIDENCE_PATTERNS = (
    re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b'),
    re.compile(r'\bgh[pousr]_[A-Za-z0-9]{20,}\b', re.I),
    re.compile(r'\bAIza[A-Za-z0-9_-]{30,}\b'),
    re.compile(r'\bAuthorization\s*:\s*Bearer\s+[A-Za-z0-9._~-]{16,}', re.I),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{20,}\b'),
)
_ASSIGNMENT_PATTERN = re.compile(
    r'\b(?:OPENAI_API_KEY|GOOGLE_API_KEY|GEMINI_API_KEY|SLACK_TOKEN|'
    r'PASSWORD|PASSWD|API_KEY|ACCESS_TOKEN|REFRESH_TOKEN|CONNECTOR_TOKEN|'
    r'CLIENT_SECRET|WEBHOOK_SECRET|GITHUB_TOKEN)\s*[:=]\s*["\']?([^\s"\']{8,})',
    re.I,
)
_CONTEXT_ASSIGNMENT_PATTERN = re.compile(
    r'\b(?P<label>[A-Za-z][A-Za-z0-9_.-]{0,95})\s*[:=]\s*'
    r'["\']?(?P<value>[^\s"\']{8,256})',
    re.I,
)
_CREDENTIAL_CONTEXT_TERMS = (
    'api_key',
    'apikey',
    'password',
    'passwd',
    'secret',
    'token',
)
_DOCUMENTED_FAKE_VALUES = frozenset(
    {
        'example',
        'example-not-a-real-key',
        'placeholder',
        'dummy',
        'test-key',
        'sk-proj-example-placeholder-not-real',
        'sk-example-placeholder-not-real',
    }
)
_HIGH_RISK_LANGUAGE_PATTERNS = (
    re.compile(r'\b(?:might|may|could|probably|possibly|proposal|proposed|opinion)\b', re.I),
    re.compile(r'\b(?:if|unless|when)\b.{0,80}\b(?:will|would|should)\b', re.I),
    re.compile(r'(?:제안|예정|추정|의견|가능성|검토\s*중|논의\s*중)'),
    re.compile(r'(?:만약|경우).{0,40}(?:예정|할\s*수|해야)'),
)


@dataclass(frozen=True)
class InputSafetyDecision:
    allowed: bool
    reason_code: str | None
    scanner_version: str = CREDENTIAL_SCAN_VERSION


def scan_auto_review_plaintext(value: str) -> InputSafetyDecision:
    for match in _ASSIGNMENT_PATTERN.finditer(value):
        if not _is_documented_fake(match.group(1)):
            return InputSafetyDecision(False, 'sensitive_input_detected')
    for match in _CONTEXT_ASSIGNMENT_PATTERN.finditer(value):
        label = re.sub(r'[^a-z0-9]+', '_', match.group('label').casefold())
        candidate = match.group('value')
        if (
            _is_credential_context_label(label)
            and not _is_documented_fake(candidate)
            and _has_bounded_credential_entropy(candidate)
        ):
            return InputSafetyDecision(False, 'sensitive_input_detected')
    for pattern in _HIGH_CONFIDENCE_PATTERNS:
        for match in pattern.finditer(value):
            if _is_documented_fake(match.group(0)):
                continue
            return InputSafetyDecision(False, 'sensitive_input_detected')
    return InputSafetyDecision(True, None)


def _is_documented_fake(value: str) -> bool:
    normalized = value.casefold().strip('"\'')
    return normalized in _DOCUMENTED_FAKE_VALUES


def _is_credential_context_label(label: str) -> bool:
    return label in _CREDENTIAL_CONTEXT_TERMS or label.endswith(
        tuple(f'_{term}' for term in _CREDENTIAL_CONTEXT_TERMS)
    )


def _has_bounded_credential_entropy(value: str) -> bool:
    candidate = value[:256]
    if len(candidate) < 16 or len(set(candidate)) < 8:
        return False
    categories = sum(
        any(predicate(character) for character in candidate)
        for predicate in (
            str.islower,
            str.isupper,
            str.isdigit,
            lambda character: not character.isalnum(),
        )
    )
    if categories < 2:
        return False
    counts = Counter(candidate)
    entropy = -sum(
        (count / len(candidate)) * math.log2(count / len(candidate))
        for count in counts.values()
    )
    return entropy >= 3.0


def contains_high_risk_language(value: str) -> bool:
    return any(pattern.search(value) for pattern in _HIGH_RISK_LANGUAGE_PATTERNS)


def provider_logging_is_safe() -> bool:
    return (
        get_debug() is False
        and os.getenv('OPENAI_LOG', '').strip().casefold() != 'debug'
        and logging.getLogger('openai').getEffectiveLevel() > logging.DEBUG
    )
