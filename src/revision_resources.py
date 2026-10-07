"""Safe cross-process resource locks for the revision queues and BAF fitting.

Locks coordinate resource use only; they never terminate another process or
change training seeds, parameters, schedules, or scientific selections.
"""

from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_DIRECTORY = PROJECT_ROOT / "results_revision" / ".resource_locks"


@contextmanager
def keep_system_awake():
    """Prevent Windows automatic sleep until this thread leaves the context.

    The request is reversible and thread-local. It does not keep the display
    awake, alter the power plan, or prevent an explicit user shutdown. Other
    operating systems use a safe no-op rather than an unverified power command.
    """
    if os.name != "nt":
        yield
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    set_execution_state = kernel.SetThreadExecutionState
    set_execution_state.argtypes = [wintypes.DWORD]
    set_execution_state.restype = wintypes.DWORD
    continuous = 0x80000000
    system_required = 0x00000001
    if not set_execution_state(continuous | system_required):
        raise RuntimeError("Cannot establish the reversible Windows system-awake request.")
    print("SYSTEM AWAKE requested while the analysis queue is active; display policy unchanged", flush=True)
    try:
        yield
    finally:
        if not set_execution_state(continuous):
            raise RuntimeError("Cannot restore the Windows thread execution-state request.")


def _unlink_with_retry(path):
    """Tolerate brief antivirus locks when releasing an explicitly owned file."""
    for attempt in range(12):
        try:
            path.unlink()
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(0.05 * (2 ** attempt), 1.0))


def process_identity(pid):
    """Return liveness and creation identity without sending a process signal.

    On Windows, ``os.kill(pid, 0)`` is deliberately not used: its semantics are
    not the POSIX liveness check. An inaccessible process is treated as unknown,
    never as a stale lock that can safely be removed.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
        return {"status": "unknown", "creation_token": None}
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            error = ctypes.get_last_error()
            return {"status": "dead" if error == 87 else "unknown", "creation_token": None}
        try:
            exit_code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return {"status": "unknown", "creation_token": None}
            if exit_code.value != 259:  # STILL_ACTIVE
                return {"status": "dead", "creation_token": None}
            creation, exit_time, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                return {"status": "unknown", "creation_token": None}
            token = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
            return {"status": "alive", "creation_token": token}
        finally:
            kernel.CloseHandle(handle)
    try:
        import psutil
        process = psutil.Process(pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return {"status": "dead", "creation_token": None}
        return {"status": "alive", "creation_token": process.create_time()}
    except ImportError:
        return {"status": "unknown", "creation_token": None}
    except psutil.NoSuchProcess:
        return {"status": "dead", "creation_token": None}
    except psutil.AccessDenied:
        return {"status": "unknown", "creation_token": None}


def owner_is_stale(owner):
    """Require positive evidence of death or PID reuse before recovering a lock."""
    current = process_identity(owner.get("pid"))
    stale_parent = (current["status"] == "dead" or
                    (current["status"] == "alive" and owner.get("creation_token") is not None
                     and current["creation_token"] != owner["creation_token"]))
    if not stale_parent:
        return False
    # If a managed parent died while its scientific child continued, the lock
    # remains reserved until that child also terminates. Do not duplicate fits.
    if owner.get("child_pid"):
        child = process_identity(owner["child_pid"])
        return (child["status"] == "dead" or
                (child["status"] == "alive" and owner.get("child_creation_token") is not None
                 and child["creation_token"] != owner["child_creation_token"]))
    return True


def queue_resource_name(results_root, lane):
    digest = hashlib.sha256(str(Path(results_root).resolve()).casefold().encode("utf-8")).hexdigest()[:16]
    return f"queue_{lane}_{digest}"


@contextmanager
def exclusive_resource(results_root, resource, description, *, wait=True,
                       poll_seconds=20, lock_directory=None):
    """Acquire one resource, waiting without killing its verified live owner.

    All BAF children use the same workspace-wide ``baf_training_ram`` resource,
    including sensitivity outputs in a different results subdirectory. Queues
    acquire a separate lane lock only, so they do not hold the RAM lock while
    waiting for a child that needs it.
    """
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", resource):
        raise ValueError("Resource names must be simple identifiers.")
    if poll_seconds <= 0:
        raise ValueError("Resource polling intervals must be positive.")
    results_root = Path(results_root).resolve()
    archive = (PROJECT_ROOT / "results").resolve()
    if results_root == archive or archive in results_root.parents:
        raise ValueError("Resource metadata must not modify the historical archive.")
    directory = Path(lock_directory or LOCK_DIRECTORY).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{resource}.lock.json"
    identity = process_identity(os.getpid())
    if identity["status"] != "alive" or identity["creation_token"] is None:
        raise RuntimeError("Cannot safely establish the current process identity for a resource lock.")
    owner = {"pid": os.getpid(), "creation_token": identity["creation_token"],
             "token": uuid.uuid4().hex, "resource": resource, "description": description,
             "results_root": str(results_root), "lock_path": str(path),
             "created_at_utc": datetime.now(timezone.utc).isoformat()}
    reported = None
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(owner, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            break
        except FileExistsError:
            try:
                previous_bytes = path.read_bytes()
                previous = json.loads(previous_bytes)
            except (FileNotFoundError, json.JSONDecodeError):
                # Another owner may be in the brief exclusive-create/write window.
                if not wait:
                    raise RuntimeError(f"Resource lock exists but is not yet verifiable: {path}")
                time.sleep(min(poll_seconds, 1))
                continue
            if previous.get("resource") != resource:
                raise RuntimeError(f"Resource lock metadata is inconsistent: {path}")
            if previous.get("pid") == owner["pid"] and previous.get("creation_token") == owner["creation_token"]:
                raise RuntimeError(f"Nested acquisition would deadlock resource {resource}.")
            if owner_is_stale(previous):
                # Serialise stale-owner recovery, including the compare/unlink
                # window. Otherwise two recovering contenders could remove a
                # newly acquired successor lock after both observed the old one.
                recovery = path.with_name(path.name + ".recovery")
                try:
                    recovery_fd = os.open(recovery, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    if not wait:
                        raise RuntimeError(f"Stale-resource recovery is already in progress: {recovery}")
                    time.sleep(min(poll_seconds, 1))
                    continue
                try:
                    with os.fdopen(recovery_fd, "w", encoding="utf-8") as stream:
                        json.dump(owner, stream)
                    try:
                        if path.read_bytes() == previous_bytes:
                            _unlink_with_retry(path)
                    except FileNotFoundError:
                        pass
                finally:
                    _unlink_with_retry(recovery)
                continue
            if not wait:
                raise RuntimeError(f"Resource {resource} is already owned by PID {previous.get('pid')}.")
            if previous.get("token") != reported:
                print(f"WAIT resource {resource}; owner PID={previous.get('pid')}; {previous.get('description')}", flush=True)
                reported = previous.get("token")
            time.sleep(poll_seconds)
    print(f"ACQUIRED resource {resource}; PID={owner['pid']}; {description}", flush=True)
    try:
        yield owner
    finally:
        keep_for_child = False
        if owner.get("child_pid"):
            child = process_identity(owner["child_pid"])
            keep_for_child = (child["status"] == "unknown" or
                              (child["status"] == "alive" and
                               (owner.get("child_creation_token") is None
                                or child["creation_token"] == owner["child_creation_token"])))
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("token") != owner["token"]:
                raise RuntimeError("Resource ownership changed unexpectedly; refusing to remove another lock.")
            if keep_for_child:
                print(f"RETAIN resource {resource}; child PID={owner['child_pid']} may still be running", flush=True)
            else:
                _unlink_with_retry(path)
        except FileNotFoundError:
            raise RuntimeError("The owned resource lock disappeared unexpectedly.")


def baf_training_resource(dataset, results_root, description):
    """Acquire the common RAM resource before a BAF CLI loads data or fits."""
    if dataset.startswith("baf"):
        return exclusive_resource(results_root, "baf_training_ram", description)
    return nullcontext()


def record_resource_child(owner, pid=None):
    """Keep a managed child reserved if its parent queue unexpectedly exits."""
    path = Path(owner["lock_path"])
    current = json.loads(path.read_text(encoding="utf-8"))
    if current.get("token") != owner["token"]:
        raise RuntimeError("Cannot record a child under a resource owned by another process.")
    if pid is None:
        owner.pop("child_pid", None)
        owner.pop("child_creation_token", None)
    else:
        owner["child_pid"] = pid
        owner["child_creation_token"] = process_identity(pid)["creation_token"]
    temporary = path.with_name(path.name + f".{owner['token']}.tmp")
    temporary.write_text(json.dumps(owner, indent=2), encoding="utf-8")
    for attempt in range(12):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(0.05 * (2 ** attempt), 1.0))
