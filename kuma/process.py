"""Stop a syntax-check subprocess and its children."""
import os
import signal
import subprocess

def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate a child started with start_new_session=True on POSIX.

    Callers must not pass a process that shares their own process group.
    Windows retains the existing direct-child-only termination behavior.
    """
    if os.name == "nt":
        process.kill()
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        pass
    # The leader can exit on SIGTERM while a descendant ignores it. Reap the
    # whole group even when waiting for the leader completed successfully.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
