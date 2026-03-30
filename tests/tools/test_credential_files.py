"""Tests for credential file passthrough and skills directory mounting."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.credential_files import (
    clear_credential_files,
    get_credential_file_mounts,
    get_skills_directory_mount,
    register_credential_file,
    register_credential_files,
    reset_config_cache,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset module state between tests."""
    clear_credential_files()
    reset_config_cache()
    yield
    clear_credential_files()
    reset_config_cache()


class TestRegisterCredentialFiles:
    def test_dict_with_path_key(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "token.json").write_text("{}")

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            missing = register_credential_files([{"path": "token.json"}])

        assert missing == []
        mounts = get_credential_file_mounts()
        assert len(mounts) == 1
        assert mounts[0]["host_path"] == str(hermes_home / "token.json")
        assert mounts[0]["container_path"] == "/root/.hermes/token.json"

    def test_dict_with_name_key_fallback(self, tmp_path):
        """Skills use 'name' instead of 'path' — both should work."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "google_token.json").write_text("{}")

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            missing = register_credential_files([
                {"name": "google_token.json", "description": "OAuth token"},
            ])

        assert missing == []
        mounts = get_credential_file_mounts()
        assert len(mounts) == 1
        assert "google_token.json" in mounts[0]["container_path"]

    def test_string_entry(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "secret.key").write_text("key")

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            missing = register_credential_files(["secret.key"])

        assert missing == []
        mounts = get_credential_file_mounts()
        assert len(mounts) == 1

    def test_missing_file_reported(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            missing = register_credential_files([
                {"name": "does_not_exist.json"},
            ])

        assert "does_not_exist.json" in missing
        assert get_credential_file_mounts() == []

    def test_path_takes_precedence_over_name(self, tmp_path):
        """When both path and name are present, path wins."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "real.json").write_text("{}")

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            missing = register_credential_files([
                {"path": "real.json", "name": "wrong.json"},
            ])

        assert missing == []
        mounts = get_credential_file_mounts()
        assert "real.json" in mounts[0]["container_path"]


class TestSkillsDirectoryMount:
    def test_returns_mount_when_skills_dir_exists(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        skills_dir = hermes_home / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "test-skill").mkdir()
        (skills_dir / "test-skill" / "SKILL.md").write_text("# test")

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            mount = get_skills_directory_mount()

        assert mount is not None
        assert mount["host_path"] == str(skills_dir)
        assert mount["container_path"] == "/root/.hermes/skills"

    def test_returns_none_when_no_skills_dir(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            mount = get_skills_directory_mount()

        assert mount is None

    def test_custom_container_base(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "skills").mkdir(parents=True)

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            mount = get_skills_directory_mount(container_base="/home/user/.hermes")

        assert mount["container_path"] == "/home/user/.hermes/skills"
