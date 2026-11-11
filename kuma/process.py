"""Stop a syntax-check subprocess and its children."""
import os
import signal
import subprocess

def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate the process and children started for the testcase."""
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
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
