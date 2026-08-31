from __future__ import annotations

import unicodedata

import pytest

from backend.app.agent_runtime.rag_v2_identity import (
    RagPublicCitationUrlValidator,
    StrictUnicodeScalarValidator,
    exact_utf8_bytes,
)


@pytest.mark.parametrize('value', ('\ud800', '\udc00', '\udc00\ud800', 'before\x00after'))
def test_strict_unicode_validator_rejects_surrogates_and_nul(value: str) -> None:
    with pytest.raises(ValueError):
        StrictUnicodeScalarValidator.validate(value)


def test_exact_utf8_bytes_preserves_non_bmp_scalar_bytes() -> None:
    value = 'A😀'

    assert exact_utf8_bytes(value) == {
        'byte_length': len(value.encode('utf-8')),
        'utf8_hex': value.encode('utf-8').hex(),
    }
    assert len(value) == 2


def test_normalized_copy_can_match_without_changing_original_bytes() -> None:
    composed = '담당자 김하나'
    decomposed = unicodedata.normalize('NFD', composed)

    assert StrictUnicodeScalarValidator.validate(composed) == composed
    assert StrictUnicodeScalarValidator.validate(decomposed) == decomposed
    assert composed.encode('utf-8') != decomposed.encode('utf-8')


def test_public_citation_url_accepts_absolute_http_urls_without_rewriting() -> None:
    validator = RagPublicCitationUrlValidator()
    value = 'HTTPS://example.test/path?q=1#part'

    assert validator.validate(value) == value


@pytest.mark.parametrize('control', ('\x7f', '\x80', '\x9f'))
def test_public_citation_url_rejects_unicode_control_scalars(control: str) -> None:
    with pytest.raises(ValueError):
        RagPublicCitationUrlValidator().validate(f'https://example.test/{control}')


@pytest.mark.parametrize(
    'value',
    (
        'https:///path',
        'https://user:pass@example.test/path',
        'https://example.test/has space',
        'https://example.test/has\nnewline',
        '//example.test/path',
        'javascript:alert(1)',
        'data:text/plain,hello',
        'file:///tmp/secret',
        'https://example.test:99999/path',
        'https://example.test\\ambiguous',
    ),
)
def test_public_citation_url_rejects_unsafe_or_ambiguous_urls(value: str) -> None:
    with pytest.raises(ValueError):
        RagPublicCitationUrlValidator().validate(value)
