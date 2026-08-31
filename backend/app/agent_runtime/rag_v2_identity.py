from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol
from urllib.parse import urlsplit

from backend.app.agent_runtime.fingerprints import (
    fingerprint_secret_bytes,
    keyed_fingerprint,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from backend.app.core.config import Settings
    from backend.app.core.demo_auth import DemoUser


_PERMISSION_LEVELS = ('public', 'internal', 'restricted')
_SECURITY_SCOPE_CONTRACT_VERSION = 'rag-security-scope:v1'
_AUTH_POLICY_VERSION = 'demo-auth:v1'
_PERMISSION_POLICY_VERSION = 'rag-permission-policy:v1'


class StrictUnicodeScalarValidator:
    @staticmethod
    def validate(value: str) -> str:
        if type(value) is not str:
            raise ValueError('text must be a string')
        for character in value:
            code_point = ord(character)
            if code_point == 0 or 0xD800 <= code_point <= 0xDFFF:
                raise ValueError('text contains an unsupported Unicode scalar')
        try:
            value.encode('utf-8', errors='strict')
        except UnicodeEncodeError as exc:
            raise ValueError('text cannot be encoded as strict UTF-8') from exc
        return value


def exact_utf8_bytes(value: str) -> dict[str, int | str]:
    encoded = StrictUnicodeScalarValidator.validate(value).encode('utf-8', errors='strict')
    return {'byte_length': len(encoded), 'utf8_hex': encoded.hex()}


class RagPublicCitationUrlValidator:
    policy_version = 'rag-public-citation-url:v1'

    def validate(self, value: str) -> str:
        value = StrictUnicodeScalarValidator.validate(value)
        if not value or any(character.isspace() or ord(character) <= 0x1F for character in value):
            raise ValueError('citation URL is invalid')
        if '\\' in value:
            raise ValueError('citation URL is ambiguous')
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as exc:
            raise ValueError('citation URL is invalid') from exc
        if (
            parsed.scheme.casefold() not in {'http', 'https'}
            or not parsed.netloc
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or port is not None and not 1 <= port <= 65535
        ):
            raise ValueError('citation URL is invalid')
        return value


@dataclass(frozen=True, slots=True)
class SecurityScope:
    contract_version: Literal['rag-security-scope:v1']
    principal_subject: str
    workspace_scope_id: str
    resource_scope_mode: Literal['all_current_scope', 'constrained']
    project_constraints: tuple[str, ...]
    source_constraints: tuple[str, ...]
    allowed_permission_levels: tuple[Literal['public', 'internal', 'restricted'], ...]
    auth_policy_version: str
    permission_policy_version: str

    def __post_init__(self) -> None:
        if self.contract_version != _SECURITY_SCOPE_CONTRACT_VERSION:
            raise ValueError('security scope contract version is invalid')
        _require_nonblank_exact(self.principal_subject)
        _require_nonblank_exact(self.workspace_scope_id)
        _require_nonblank_exact(self.auth_policy_version)
        _require_nonblank_exact(self.permission_policy_version)
        if self.resource_scope_mode not in {'all_current_scope', 'constrained'}:
            raise ValueError('security scope mode is invalid')
        _validate_project_constraints(self.project_constraints)
        _validate_source_constraints(self.source_constraints)
        _validate_permission_levels(self.allowed_permission_levels)
        is_empty = not self.project_constraints and not self.source_constraints
        if (self.resource_scope_mode == 'all_current_scope') != is_empty:
            raise ValueError('security scope mode and constraints do not match')


class RagSecurityScopeResolver(Protocol):
    def resolve(self, *, db: Session, actor: DemoUser) -> SecurityScope: ...


@dataclass(frozen=True, slots=True)
class ServerRagSecurityScopeResolver:
    """Builds the current deployment's all-current-scope actor boundary."""

    settings: Settings

    def resolve(self, *, db: Session, actor: DemoUser) -> SecurityScope:
        del db
        permissions = tuple(
            level for level in _PERMISSION_LEVELS if level in actor.permission_levels
        )
        return SecurityScope(
            contract_version=_SECURITY_SCOPE_CONTRACT_VERSION,
            principal_subject=actor.id,
            workspace_scope_id=self.settings.agent_runtime_security_scope_id,
            resource_scope_mode='all_current_scope',
            project_constraints=(),
            source_constraints=(),
            allowed_permission_levels=permissions,
            auth_policy_version=_AUTH_POLICY_VERSION,
            permission_policy_version=_PERMISSION_POLICY_VERSION,
        )


def security_scope_fingerprint(scope: SecurityScope, *, settings: Settings) -> str:
    key, _ = fingerprint_secret_bytes(settings)
    return keyed_fingerprint(
        {
            'contract_version': scope.contract_version,
            'principal_subject_bytes': exact_utf8_bytes(scope.principal_subject),
            'workspace_scope_id_bytes': exact_utf8_bytes(scope.workspace_scope_id),
            'resource_scope_mode': scope.resource_scope_mode,
            'project_constraint_bytes': [
                exact_utf8_bytes(value) for value in scope.project_constraints
            ],
            'source_constraint_bytes': [
                exact_utf8_bytes(value) for value in scope.source_constraints
            ],
            'allowed_permission_levels': list(scope.allowed_permission_levels),
            'auth_policy_version_bytes': exact_utf8_bytes(scope.auth_policy_version),
            'permission_policy_version_bytes': exact_utf8_bytes(
                scope.permission_policy_version
            ),
        },
        secret=key,
        schema_version='rag-security-scope-fingerprint:v1',
        policy_version=_SECURITY_SCOPE_CONTRACT_VERSION,
    )


def _require_nonblank_exact(value: str) -> None:
    StrictUnicodeScalarValidator.validate(value)
    if not value.strip() or value != value.strip():
        raise ValueError('security scope string is invalid')


def _validate_project_constraints(values: tuple[str, ...]) -> None:
    if type(values) is not tuple:
        raise ValueError('project constraints are invalid')
    normalized: list[str] = []
    for value in values:
        StrictUnicodeScalarValidator.validate(value)
        if not value.startswith('project_key:') or not value.removeprefix('project_key:'):
            raise ValueError('project constraint is invalid')
        if value != f"project_key:{unicodedata.normalize('NFC', value.removeprefix('project_key:'))}":
            raise ValueError('project constraint is not canonical')
        normalized.append(value)
    if tuple(sorted(normalized)) != values or len(set(values)) != len(values):
        raise ValueError('project constraints are not ordered')


def _validate_source_constraints(values: tuple[str, ...]) -> None:
    if type(values) is not tuple:
        raise ValueError('source constraints are invalid')
    identifiers: list[int] = []
    for value in values:
        StrictUnicodeScalarValidator.validate(value)
        if not value.startswith('source_pk:'):
            raise ValueError('source constraint is invalid')
        raw_identifier = value.removeprefix('source_pk:')
        if not raw_identifier.isascii() or not raw_identifier.isdecimal():
            raise ValueError('source constraint is invalid')
        identifier = int(raw_identifier)
        if identifier <= 0 or str(identifier) != raw_identifier:
            raise ValueError('source constraint is invalid')
        identifiers.append(identifier)
    if identifiers != sorted(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError('source constraints are not ordered')


def _validate_permission_levels(
    values: tuple[Literal['public', 'internal', 'restricted'], ...],
) -> None:
    if type(values) is not tuple or any(value not in _PERMISSION_LEVELS for value in values):
        raise ValueError('security scope permissions are invalid')
    expected = tuple(value for value in _PERMISSION_LEVELS if value in values)
    if values != expected:
        raise ValueError('security scope permissions are not ordered')
