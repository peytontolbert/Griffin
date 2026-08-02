"""Tests for installable project metadata and optional dependency groups."""

from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.10 uses the test extra's compatibility parser.
    import tomli as tomllib


def test_pyproject_declares_runtime_and_development_dependencies():
    """Project metadata should support core, training, testing, and Triton installs."""
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]

    assert project["name"] == "griffin-language-model"
    assert any(requirement.startswith("torch") for requirement in project["dependencies"])
    assert {"train", "test", "triton", "all"} <= set(
        project["optional-dependencies"]
    )
