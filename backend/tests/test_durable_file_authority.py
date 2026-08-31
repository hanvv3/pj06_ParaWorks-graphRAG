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
