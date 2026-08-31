from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from threading import Lock, RLock

_PROCESS_LOCKS_GUARD = Lock()
_PROCESS_LOCKS: dict[str, RLock] = {}


def _process_lock(path: Path) -> RLock:
    identity = os.path.normcase(str(path.absolute()))
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(identity, RLock())


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
        text = os.fspath(path)
        if not text.strip() or '\x00' in text:
            raise DurableFileAuthorityError('authority path is invalid')
        normalized_parts = text.replace('\\', '/').split('/')
        candidate = Path(text)
        if not candidate.is_absolute() or any(
            part in {'.', '..'} for part in normalized_parts
        ):
            raise DurableFileAuthorityError('authority path must be canonical absolute')
        if os.name == 'nt' and any(
            part.endswith(('.', ' ')) for part in candidate.parts[1:]
        ):
            raise DurableFileAuthorityError('authority path has a Windows alias')
        current = candidate
        existing: list[Path] = []
        while not current.exists() and current != current.parent:
            current = current.parent
        while True:
            existing.append(current)
            if current == candidate or len(current.parts) >= len(candidate.parts):
                break
            current = Path(*candidate.parts[: len(current.parts) + 1])
        for item in existing:
            try:
                info = item.lstat()
            except FileNotFoundError:
                continue
            reparse = bool(
                getattr(info, 'st_file_attributes', 0)
                & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)
            )
            if item.is_symlink() or reparse:
                raise DurableFileAuthorityError(
                    'authority path cannot traverse a link or reparse point'
                )
            if os.name == 'nt' and item != Path(item.anchor):
                matches = [
                    child.name
                    for child in item.parent.iterdir()
                    if child.name.casefold() == item.name.casefold()
                ]
                if matches != [item.name]:
                    raise DurableFileAuthorityError(
                        'authority path has a case-fold alias'
                    )
        return candidate

    def _validate_existing_regular(self, path: Path) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(info.st_mode) or path.is_symlink():
            raise DurableFileAuthorityError('authority file must be regular')
        if info.st_nlink != 1:
            raise DurableFileAuthorityError('authority file cannot be hard-linked')
        if os.name != 'nt' and stat.S_IMODE(info.st_mode) & 0o077:
            raise DurableFileAuthorityError('authority file must be user-only')

    @staticmethod
    def _validate_parent_security(path: Path) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise DurableFileAuthorityError('authority parent is unavailable') from exc
        if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
            raise DurableFileAuthorityError('authority parent must be a directory')
        if os.name != 'nt' and stat.S_IMODE(info.st_mode) & 0o077:
            raise DurableFileAuthorityError('authority parent must be user-only')

    @staticmethod
    def _validate_open_identity(path: Path, descriptor: int) -> None:
        opened = os.fstat(descriptor)
        named = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != named.st_dev
            or opened.st_ino != named.st_ino
            or opened.st_nlink != 1
        ):
            raise DurableFileAuthorityError('authority file identity changed')

    def _read_bytes_unlocked(self) -> bytes:
        self._validate_existing_regular(self.path)
        flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0)
        flags |= getattr(os, 'O_NOFOLLOW', 0)
        try:
            descriptor = os.open(self.path, flags)
        except OSError as exc:
            raise DurableFileAuthorityError('authority envelope is unavailable') from exc
        try:
            self._validate_open_identity(self.path, descriptor)
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 64 * 1024):
                chunks.append(chunk)
            return b''.join(chunks)
        finally:
            os.close(descriptor)

    @contextmanager
    def locked(self) -> Iterator[None]:
        with _process_lock(self.lock_path), self._locked_file():
            yield

    @contextmanager
    def _locked_file(self) -> Iterator[None]:
        if self._runtime:
            if (
                not self.path.parent.is_dir()
                or not self.path.is_file()
                or not self.lock_path.is_file()
            ):
                raise DurableFileAuthorityError('runtime authority is unavailable')
        else:
            parent_existed = self.path.parent.exists()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not parent_existed:
                os.chmod(self.path.parent, 0o700)
            self.validate_configured_path(self.path)
        self._validate_parent_security(self.path.parent)
        self._validate_existing_regular(self.lock_path)
        if self._runtime or self._initialized:
            handle = self.lock_path.open('r+b')
        else:
            data_exists = self.path.exists()
            lock_exists = self.lock_path.exists()
            if data_exists and not lock_exists:
                raise DurableFileAuthorityError(
                    'authority partial initialization is unsafe'
                )
            if data_exists and lock_exists:
                raise DurableFileAuthorityError(
                    'authority initialization already exists'
                )
            if lock_exists:
                handle = self.lock_path.open('r+b')
            else:
                handle = self.lock_path.open('x+b')
                os.chmod(self.lock_path, 0o600)
            self._initialized = True
        try:
            self._validate_open_identity(self.lock_path, handle.fileno())
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
                raw = self._read_bytes_unlocked()
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
                current = json.loads(self._read_bytes_unlocked().decode('utf-8'))
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
                os.chmod(handle.name, 0o600)
            if os.name == 'nt':
                self._replace_windows_write_through(Path(temp_name), self.path)
            else:
                os.replace(temp_name, self.path)
            temp_name = None
            self._validate_existing_regular(self.path)
            self._flush_parent_directory()
        finally:
            if temp_name is not None:
                with suppress(FileNotFoundError):
                    os.unlink(temp_name)

    def _flush_parent_directory(self) -> None:
        if os.name != 'nt':
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return
        # Windows directory handles do not support FlushFileBuffers. The rename
        # itself is issued with MOVEFILE_WRITE_THROUGH below.

    @staticmethod
    def _replace_windows_write_through(source: Path, target: Path) -> None:
        import ctypes
        from ctypes import wintypes

        move_file = ctypes.windll.kernel32.MoveFileExW
        move_file.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
        move_file.restype = wintypes.BOOL
        if not move_file(str(source), str(target), 0x00000001 | 0x00000008):
            raise DurableFileAuthorityError(
                'authority durable replace is unsupported'
            )
