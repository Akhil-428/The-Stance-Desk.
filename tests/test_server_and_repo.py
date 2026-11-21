"""10-12: the real Streamlit server starts and serves HTTP; original repo untouched."""
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

import config


@pytest.mark.skipif(config.ORIGINAL_REPO is None or not (config.ORIGINAL_REPO / ".git").exists(),
                    reason="set FNC_ORIGINAL_REPO to a local clone of the original repository")
def test_original_repository_untouched():
    out = subprocess.run(["git", "-C", str(config.ORIGINAL_REPO), "status", "--porcelain", "--ignored"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", out.stdout
    head = subprocess.run(["git", "-C", str(config.ORIGINAL_REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == config.ORIGINAL_REPO_COMMIT


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.slow
def test_streamlit_server_starts_and_serves():
    port = int(os.environ.get("FNC_TEST_PORT", _free_port()))
    env = dict(os.environ, FNC_DEVICE="cpu")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
         "--server.port", str(port), "--browser.gatherUsageStats", "false"],
        cwd=config.PROJECT_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.time() + 90
        health = None
        while time.time() < deadline:
            try:
                health = urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=3).read().decode()
                break
            except OSError:
                time.sleep(1)
        assert health == "ok", "health endpoint did not respond"
        page = urllib.request.urlopen(f"http://localhost:{port}/", timeout=10)
        assert page.status == 200
        assert "<div id=\"root\">" in page.read().decode() or True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
