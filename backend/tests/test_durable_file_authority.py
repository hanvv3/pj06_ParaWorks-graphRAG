from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backend.app.agent_runtime.durable_file_authority import (
    DurableFileAuthority,
    DurableFileAuthorityError,
)


def test_stable_lock_inode_is_never_replaced_and_data_is_atomic(tmp_path: Path):
    data = tmp_path / 'provider-safety.json'
    authority = DurableFileAuthority(data)
    authority.write({'generation': 1, 'hmac': 'a' * 64})
    lock = Path(str(data) + '.lock')
    inode = os.stat(lock).st_ino
    authority.write({'generation': 2, 'hmac': 'b' * 64})
    assert os.stat(lock).st_ino == inode
    assert authority.read() == {'generation': 2, 'hmac': 'b' * 64}
    assert json.loads(data.read_text(encoding='utf-8'))['generation'] == 2


def test_disabled_validation_creates_no_artifacts(tmp_path: Path):
    data = tmp_path / 'disabled.json'
    DurableFileAuthority.validate_configured_path(data)
    assert not data.exists()
    assert not Path(str(data) + '.lock').exists()


def test_symlink_and_hardlink_authority_are_rejected(tmp_path: Path):
    target = tmp_path / 'target.json'
    target.write_text('{}', encoding='utf-8')
    alias = tmp_path / 'alias.json'
    try:
        alias.symlink_to(target)
    except OSError:
        pytest.skip('symlink creation unavailable')
    with pytest.raises(DurableFileAuthorityError):
        DurableFileAuthority(alias).read()

    hardlink = tmp_path / 'hardlink.json'
    os.link(target, hardlink)
    with pytest.raises(DurableFileAuthorityError):
        DurableFileAuthority(target).read()


def test_runtime_read_of_missing_authority_creates_no_parent_or_lock(tmp_path: Path):
    data = tmp_path / 'missing-parent' / 'provider.json'
    with pytest.raises(DurableFileAuthorityError):
        DurableFileAuthority.open_runtime(data).read()
    assert not data.parent.exists()


def test_secure_init_reuses_trusted_sidecar_alone_but_refuses_partial_data(
    tmp_path: Path,
):
    data = tmp_path / 'provider.json'
    lock = Path(str(data) + '.lock')
    lock.write_bytes(b'\0')
    os.chmod(lock, 0o600)
    inode = os.stat(lock).st_ino
    DurableFileAuthority(data).write({'generation': 0})
    assert os.stat(lock).st_ino == inode

    partial = tmp_path / 'partial.json'
    partial.write_text('{}', encoding='utf-8')
    os.chmod(partial, 0o600)
    with pytest.raises(DurableFileAuthorityError, match='partial'):
        DurableFileAuthority(partial).write({'generation': 0})


def test_config_rejects_relative_dot_alias_and_symlinked_parent(tmp_path: Path):
    with pytest.raises(DurableFileAuthorityError):
        DurableFileAuthority.validate_configured_path('relative/provider.json')
    with pytest.raises(DurableFileAuthorityError):
        DurableFileAuthority.validate_configured_path(
            str(tmp_path) + os.sep + '.' + os.sep + 'provider.json'
        )

    real_parent = tmp_path / 'real'
    real_parent.mkdir()
    alias_parent = tmp_path / 'alias-parent'
    try:
        alias_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip('directory symlink creation unavailable')
    with pytest.raises(DurableFileAuthorityError):
        DurableFileAuthority.validate_configured_path(alias_parent / 'provider.json')


def test_replace_invokes_platform_durability_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data = tmp_path / 'provider.json'
    authority = DurableFileAuthority(data)
    flushed: list[Path] = []
    monkeypatch.setattr(
        authority,
        '_flush_parent_directory',
        lambda: flushed.append(data.parent),
    )
    authority.write({'generation': 0})
    assert flushed == [data.parent]


def test_lock_is_released_after_exception(tmp_path: Path):
    authority = DurableFileAuthority(tmp_path / 'provider.json')
    with (
        pytest.raises(RuntimeError, match='simulated crash boundary'),
        authority.locked(),
    ):
        raise RuntimeError('simulated crash boundary')
    with authority.locked():
        pass


@pytest.mark.skipif(os.name != 'nt', reason='Windows write-through primitive')
def test_windows_replace_uses_write_through_primitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data = tmp_path / 'provider.json'
    authority = DurableFileAuthority(data)
    original = authority._replace_windows_write_through
    replacements: list[tuple[Path, Path]] = []

    def observed(source: Path, target: Path) -> None:
        replacements.append((source, target))
        original(source, target)

    monkeypatch.setattr(authority, '_replace_windows_write_through', observed)
    authority.write({'generation': 0})
    assert len(replacements) == 1
    assert replacements[0][1] == data


@pytest.mark.skipif(os.name != 'nt', reason='Windows path alias semantics')
def test_windows_casefold_and_trailing_dot_aliases_are_rejected(tmp_path: Path):
    (tmp_path / 'Provider.JSON').write_text('{}', encoding='utf-8')
    with pytest.raises(DurableFileAuthorityError, match='case-fold'):
        DurableFileAuthority.validate_configured_path(tmp_path / 'provider.json')
    with pytest.raises(DurableFileAuthorityError, match='Windows alias'):
        DurableFileAuthority.validate_configured_path(
            str(tmp_path / 'provider.json') + '.'
        )


@pytest.mark.skipif(os.name == 'nt', reason='POSIX permission mode assertion')
def test_runtime_rejects_group_or_other_access_on_authority_files(tmp_path: Path):
    data = tmp_path / 'provider.json'
    authority = DurableFileAuthority(data)
    authority.write({'generation': 0})
    os.chmod(Path(str(data) + '.lock'), 0o644)
    with pytest.raises(DurableFileAuthorityError, match='user-only'):
        DurableFileAuthority.open_runtime(data).read()


@pytest.mark.skipif(os.name == 'nt', reason='POSIX permission mode assertion')
def test_runtime_rejects_group_or_other_access_on_authority_parent(tmp_path: Path):
    protected = tmp_path / 'protected'
    data = protected / 'provider.json'
    authority = DurableFileAuthority(data)
    authority.write({'generation': 0})
    os.chmod(protected, 0o755)
    with pytest.raises(DurableFileAuthorityError, match='parent must be user-only'):
        DurableFileAuthority.open_runtime(data).read()
