from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PLACEHOLDERS = ('xoxb-test', '<OPENAI_API_KEY>', 'example', 'changeme')
_KNOWN_TEST_FIXTURE_DETECTORS = {
    ('backend/tests/test_auto_review_eligibility.py', 'private_key'),
    ('backend/tests/test_integration_runtime_status.py', 'slack_token'),
    ('backend/tests/test_review_v21_extraction.py', 'openai_key'),
}
_DETECTORS = (
    ('openai_key', re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b')),
    ('slack_token', re.compile(rb'\bxox[baprs]-[A-Za-z0-9-]{20,}\b')),
    ('github_token', re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b')),
    ('private_key', re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
    (
        'password_assignment',
        re.compile(rb'(?i)\b(?:password|passwd)\s*[:=]\s*["\'][^"\'\r\n]{12,}["\']'),
    ),
)


@dataclass(frozen=True, slots=True)
class SecretFinding:
    relative_path: str
    line_number: int
    detector_kind: str

    def __str__(self) -> str:
        return f'{self.relative_path}:{self.line_number}:{self.detector_kind}'


def _release_files() -> tuple[Path, ...]:
    result = subprocess.run(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    names = tuple(name for name in result.stdout.split(b'\0') if name)
    return tuple(REPOSITORY_ROOT / name.decode('utf-8') for name in names)


def scan_release_files() -> tuple[SecretFinding, ...]:
    findings: list[SecretFinding] = []
    for path in _release_files():
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            lines = path.read_bytes().splitlines()
        except OSError:
            continue
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        for line_number, line in enumerate(lines, start=1):
            lowered = line.lower()
            if any(marker.lower().encode() in lowered for marker in _PLACEHOLDERS):
                continue
            for kind, pattern in _DETECTORS:
                if (
                    pattern.search(line)
                    and (relative, kind) not in _KNOWN_TEST_FIXTURE_DETECTORS
                ):
                    findings.append(SecretFinding(relative, line_number, kind))
    return tuple(findings)


def test_repository_has_no_high_confidence_secret_material():
    findings = scan_release_files()
    assert not findings, '\n'.join(str(finding) for finding in findings)


def test_scanner_finds_untracked_sample_without_echoing_secret_bytes():
    sample = REPOSITORY_ROOT / '.task16-secret-scanner-sample.txt'
    secret = ''.join(('gh', 'p_', 'A' * 36))
    try:
        sample.write_text(secret, encoding='utf-8')
        findings = scan_release_files()
        rendered = '\n'.join(str(finding) for finding in findings)
        assert any(
            finding.relative_path == sample.name
            and finding.detector_kind == 'github_token'
            for finding in findings
        )
        assert secret not in rendered
    finally:
        sample.unlink(missing_ok=True)


def test_documented_placeholders_are_allowlisted():
    sample = REPOSITORY_ROOT / '.task16-secret-placeholder-sample.txt'
    try:
        sample.write_text(
            'OPENAI_API_KEY=<OPENAI_API_KEY>\nSLACK_TOKEN=xoxb-test\n',
            encoding='utf-8',
        )
        assert all(
            finding.relative_path != sample.name for finding in scan_release_files()
        )
    finally:
        sample.unlink(missing_ok=True)
