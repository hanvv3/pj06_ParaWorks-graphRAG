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


def _windows_current_user_sid() -> str:
    if os.name != 'nt':
        raise DurableFileAuthorityError('Windows security API is unavailable')
    import ctypes
    from ctypes import wintypes

    class SidAndAttributes(ctypes.Structure):
        _fields_ = [('sid', wintypes.LPVOID), ('attributes', wintypes.DWORD)]

    class TokenUser(ctypes.Structure):
        _fields_ = [('user', SidAndAttributes)]

    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.argtypes = ()
    get_current_process.restype = wintypes.HANDLE
    open_process_token = ctypes.windll.advapi32.OpenProcessToken
    open_process_token.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    )
    open_process_token.restype = wintypes.BOOL
    get_token_information = ctypes.windll.advapi32.GetTokenInformation
    get_token_information.argtypes = (
        wintypes.HANDLE,
        ctypes.c_uint,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    get_token_information.restype = wintypes.BOOL
    convert_sid = ctypes.windll.advapi32.ConvertSidToStringSidW
    convert_sid.argtypes = (wintypes.LPVOID, ctypes.POINTER(wintypes.LPWSTR))
    convert_sid.restype = wintypes.BOOL
    token = wintypes.HANDLE()
    if not open_process_token(get_current_process(), 0x0008, ctypes.byref(token)):
        raise DurableFileAuthorityError('Windows security API is unavailable')
    try:
        needed = wintypes.DWORD()
        get_token_information(token, 1, None, 0, ctypes.byref(needed))
        if needed.value == 0:
            raise DurableFileAuthorityError('Windows security API is unavailable')
        buffer = ctypes.create_string_buffer(needed.value)
        if not get_token_information(
            token, 1, buffer, needed, ctypes.byref(needed)
        ):
            raise DurableFileAuthorityError('Windows security API is unavailable')
        sid = ctypes.cast(buffer, ctypes.POINTER(TokenUser)).contents.user.sid
        text = wintypes.LPWSTR()
        if not convert_sid(wintypes.LPVOID(sid), ctypes.byref(text)):
            raise DurableFileAuthorityError('Windows security API is unavailable')
        try:
            return text.value
        finally:
            ctypes.windll.kernel32.LocalFree(text)
    finally:
        ctypes.windll.kernel32.CloseHandle(token)


def _windows_apply_sddl(path: Path, sddl: str) -> None:
    if os.name != 'nt':
        raise DurableFileAuthorityError('Windows security API is unavailable')
    import ctypes
    from ctypes import wintypes

    descriptor = wintypes.LPVOID()
    descriptor_size = wintypes.DWORD()
    if not ctypes.windll.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl, 1, ctypes.byref(descriptor), ctypes.byref(descriptor_size)
    ):
        raise DurableFileAuthorityError('Windows security API is unavailable')
    try:
        security_information = 0x00000004 | 0x80000000
        if not ctypes.windll.advapi32.SetFileSecurityW(
            str(path), security_information, descriptor
        ):
            raise DurableFileAuthorityError('Windows owner-only DACL is unavailable')
    finally:
        ctypes.windll.kernel32.LocalFree(descriptor)


def _windows_apply_owner_only(path: Path) -> None:
    sid = _windows_current_user_sid()
    _windows_apply_sddl(path, f'O:{sid}D:P(A;;FA;;;{sid})')
    if not _windows_has_owner_only_dacl(path):
        raise DurableFileAuthorityError('Windows owner-only DACL is unavailable')


def _windows_has_owner_only_dacl(path: Path) -> bool:
    if os.name != 'nt':
        raise DurableFileAuthorityError('Windows security API is unavailable')
    import ctypes
    from ctypes import wintypes

    class AclSizeInformation(ctypes.Structure):
        _fields_ = [
            ('ace_count', wintypes.DWORD),
            ('acl_bytes_in_use', wintypes.DWORD),
            ('acl_bytes_free', wintypes.DWORD),
        ]

    class AceHeader(ctypes.Structure):
        _fields_ = [
            ('ace_type', ctypes.c_ubyte),
            ('ace_flags', ctypes.c_ubyte),
            ('ace_size', wintypes.WORD),
        ]

    class AccessAllowedAce(ctypes.Structure):
        _fields_ = [
            ('header', AceHeader),
            ('mask', wintypes.DWORD),
            ('sid_start', wintypes.DWORD),
        ]

    requested = 0x00000001 | 0x00000004
    needed = wintypes.DWORD()
    ctypes.windll.advapi32.GetFileSecurityW(
        str(path), requested, None, 0, ctypes.byref(needed)
    )
    if needed.value == 0:
        raise DurableFileAuthorityError('Windows security API is unavailable')
    descriptor = ctypes.create_string_buffer(needed.value)
    if not ctypes.windll.advapi32.GetFileSecurityW(
        str(path), requested, descriptor, needed, ctypes.byref(needed)
    ):
        raise DurableFileAuthorityError('Windows security API is unavailable')
    descriptor_ptr = ctypes.cast(descriptor, wintypes.LPVOID)
    owner = wintypes.LPVOID()
    owner_defaulted = wintypes.BOOL()
    if not ctypes.windll.advapi32.GetSecurityDescriptorOwner(
        descriptor_ptr, ctypes.byref(owner), ctypes.byref(owner_defaulted)
    ):
        raise DurableFileAuthorityError('Windows security API is unavailable')
    present = wintypes.BOOL()
    defaulted = wintypes.BOOL()
    acl = wintypes.LPVOID()
    if not ctypes.windll.advapi32.GetSecurityDescriptorDacl(
        descriptor_ptr,
        ctypes.byref(present),
        ctypes.byref(acl),
        ctypes.byref(defaulted),
    ):
        raise DurableFileAuthorityError('Windows security API is unavailable')
    control = wintypes.WORD()
    revision = wintypes.DWORD()
    if not ctypes.windll.advapi32.GetSecurityDescriptorControl(
        descriptor_ptr, ctypes.byref(control), ctypes.byref(revision)
    ):
        raise DurableFileAuthorityError('Windows security API is unavailable')
    if not present.value or not acl.value or not control.value & 0x1000:
        return False
    info = AclSizeInformation()
    if not ctypes.windll.advapi32.GetAclInformation(
        acl, ctypes.byref(info), ctypes.sizeof(info), 2
    ):
        raise DurableFileAuthorityError('Windows security API is unavailable')
    if info.ace_count != 1:
        return False
    ace = wintypes.LPVOID()
    if not ctypes.windll.advapi32.GetAce(acl, 0, ctypes.byref(ace)):
        raise DurableFileAuthorityError('Windows security API is unavailable')
    allowed = ctypes.cast(ace, ctypes.POINTER(AccessAllowedAce)).contents
    if allowed.header.ace_type != 0 or allowed.mask != 0x001F01FF:
        return False
    ace_sid = wintypes.LPVOID(
        ace.value + AccessAllowedAce.sid_start.offset
    )
    current_sid_text = _windows_current_user_sid()
    current_descriptor = wintypes.LPVOID()
    current_descriptor_size = wintypes.DWORD()
    if not ctypes.windll.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        f'O:{current_sid_text}',
        1,
        ctypes.byref(current_descriptor),
        ctypes.byref(current_descriptor_size),
    ):
        raise DurableFileAuthorityError('Windows security API is unavailable')
    try:
        current_owner = wintypes.LPVOID()
        if not ctypes.windll.advapi32.GetSecurityDescriptorOwner(
            current_descriptor,
            ctypes.byref(current_owner),
            ctypes.byref(owner_defaulted),
        ):
            raise DurableFileAuthorityError('Windows security API is unavailable')
        return bool(
            ctypes.windll.advapi32.EqualSid(owner, current_owner)
            and ctypes.windll.advapi32.EqualSid(ace_sid, current_owner)
        )
    finally:
        ctypes.windll.kernel32.LocalFree(current_descriptor)


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
        if os.name == 'nt':
            if not _windows_has_owner_only_dacl(path):
                raise DurableFileAuthorityError(
                    'authority file must have an owner-only DACL'
                )
        elif stat.S_IMODE(info.st_mode) & 0o077:
            raise DurableFileAuthorityError('authority file must be user-only')

    @staticmethod
    def _validate_parent_security(path: Path) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise DurableFileAuthorityError('authority parent is unavailable') from exc
        if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
            raise DurableFileAuthorityError('authority parent must be a directory')
        if os.name == 'nt':
            if not _windows_has_owner_only_dacl(path):
                raise DurableFileAuthorityError(
                    'authority parent must have an owner-only DACL'
                )
        elif stat.S_IMODE(info.st_mode) & 0o077:
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
            if not parent_existed and os.name != 'nt':
                os.chmod(self.path.parent, 0o700)
            self.validate_configured_path(self.path)
            if os.name == 'nt':
                _windows_apply_owner_only(self.path.parent)
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
                if os.name == 'nt':
                    _windows_apply_owner_only(self.lock_path)
                else:
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
                if os.name == 'nt':
                    _windows_apply_owner_only(Path(handle.name))
                else:
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
