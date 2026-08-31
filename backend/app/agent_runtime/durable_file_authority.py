from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path


class DurableFileAuthorityError(RuntimeError):
    pass


class DurableFileAuthority:
    """Stable OS-lock sidecar plus atomically replaced canonical JSON data."""

    def __init__(self, path: str | Path, *, runtime: bool = False) -> None:
        self.path = self.validate_configured_path(path)
        self.lock_path = Path(str(self.path) + '.lock')
        self._runtime = runtime
        self._initialized = runtime

    @classmethod
    def open_runtime(cls, path: str | Path) -> DurableFileAuthority:
        """Open an existing authority without creating any artifact."""
        return cls(path, runtime=True)

    @staticmethod
    def validate_configured_path(path: str | Path) -> Path:
        if type(path) is not str and not isinstance(path, os.PathLike):
            raise DurableFileAuthorityError('authority path is invalid')
        candidate = Path(path)
        text = str(candidate)
        if not text.strip() or '\x00' in text:
            raise DurableFileAuthorityError('authority path is invalid')
        if candidate.exists() and candidate.is_symlink():
            raise DurableFileAuthorityError('authority path cannot be a symlink')
        return candidate.absolute()

    def _validate_existing_regular(self, path: Path) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(info.st_mode) or path.is_symlink():
            raise DurableFileAuthorityError('authority file must be regular')
        if info.st_nlink != 1:
            raise DurableFileAuthorityError('authority file cannot be hard-linked')

    @contextmanager
    def locked(self) -> Iterator[None]:
        if self._runtime:
            if (
                not self.path.parent.is_dir()
                or not self.path.is_file()
                or not self.lock_path.is_file()
            ):
                raise DurableFileAuthorityError('runtime authority is unavailable')
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._validate_existing_regular(self.lock_path)
        if self._runtime or self._initialized:
            handle = self.lock_path.open('r+b')
        else:
            try:
                handle = self.lock_path.open('x+b')
            except FileExistsError:
                raise DurableFileAuthorityError(
                    'authority initialization already exists'
                ) from None
            self._initialized = True
        try:
            if handle.tell() == 0:
                handle.write(b'\0')
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        except OSError as exc:
            raise DurableFileAuthorityError('stable authority lock failed') from exc
        finally:
            try:
                handle.seek(0)
                if os.name == 'nt':
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def read(self) -> dict[str, object]:
        self._validate_existing_regular(self.path)
        with self.locked():
            self._validate_existing_regular(self.path)
            try:
                raw = self.path.read_bytes()
                parsed = json.loads(raw.decode('utf-8'))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise DurableFileAuthorityError('authority envelope is invalid') from exc
            if type(parsed) is not dict:
                raise DurableFileAuthorityError('authority envelope is invalid')
            return parsed

    def update(
        self,
        transform: Callable[[dict[str, object]], dict[str, object]],
        *,
        after_replace: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, object]:
        """Read/modify/replace under one stable sidecar acquisition."""
        with self.locked():
            self._validate_existing_regular(self.path)
            try:
                current = json.loads(self.path.read_text(encoding='utf-8'))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise DurableFileAuthorityError('authority envelope is invalid') from exc
            if type(current) is not dict:
                raise DurableFileAuthorityError('authority envelope is invalid')
            updated = transform(current)
            if type(updated) is not dict:
                raise DurableFileAuthorityError('authority envelope is invalid')
            self._replace_unlocked(updated)
            if after_replace is not None:
                after_replace(updated)
            return updated

    def write(self, value: dict[str, object]) -> None:
        if type(value) is not dict:
            raise DurableFileAuthorityError('authority envelope is invalid')
        with self.locked():
            self._validate_existing_regular(self.path)
            self._replace_unlocked(value)

    def _replace_unlocked(self, value: dict[str, object]) -> None:
        encoded = json.dumps(
            value, ensure_ascii=False, allow_nan=False,
            separators=(',', ':'), sort_keys=True,
        ).encode('utf-8')
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='wb', dir=self.path.parent, prefix=f'.{self.path.name}.',
                delete=False,
            ) as handle:
                temp_name = handle.name
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
            temp_name = None
            if os.name != 'nt':
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temp_name is not None:
                with suppress(FileNotFoundError):
                    os.unlink(temp_name)
