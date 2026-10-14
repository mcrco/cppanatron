"""build_info(): how the loaded extension was built."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import cppanatron as cpp
from cppanatron import _core

REPO = Path(__file__).resolve().parents[2]


def test_build_info_reports_the_build():
    info = _core.build_info()
    assert set(info) == {
        "version", "git_sha", "git_dirty", "compiler", "compiler_version", "build_type",
        "cxx_flags", "cxx_standard", "nanobind_version", "python_version",
    }
    assert info["version"] == cpp.__version__
    assert info["git_sha"] == "unknown" or re.fullmatch(r"[0-9a-f]{40}", info["git_sha"])
    assert isinstance(info["git_dirty"], bool)
    assert info["compiler"] and info["compiler_version"]
    assert info["cxx_standard"] >= 202002
    assert re.fullmatch(r"\d+\.\d+\.\d+", info["nanobind_version"])
    import platform

    assert info["python_version"].startswith(".".join(platform.python_version_tuple()[:2]))


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_recorded_commit_is_in_this_checkout():
    sha = _core.build_info()["git_sha"]
    if sha == "unknown":
        pytest.skip("not built from a git checkout")
    result = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", sha, "HEAD"],
        capture_output=True,
    )
    assert result.returncode == 0, f"extension built from {sha}, not an ancestor of HEAD"
