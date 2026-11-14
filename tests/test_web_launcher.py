"""A ready response must not make the launcher reuse an outdated backend."""

from fastapi.testclient import TestClient
import pytest

from ml_opt.web import __main__ as launcher
from ml_opt.web.app import create_app
from ml_opt.web.runtime import source_revision


def test_revision_changes_with_source_and_frontend_but_ignores_cache(tmp_path):
    module = tmp_path / "app.py"
    module.write_text("old code")
    first = source_revision(tmp_path)
    module.write_text("updated code")
    assert source_revision(tmp_path) != first
    second = source_revision(tmp_path)
    (tmp_path / "app.js").write_text("new frontend")
    assert source_revision(tmp_path) != second
    third = source_revision(tmp_path)
    (tmp_path / "app.pyc").write_bytes(b"compiled")
    assert source_revision(tmp_path) == third


def test_health_reports_revision_captured_at_app_creation(monkeypatch):
    monkeypatch.setattr("ml_opt.web.app.source_revision", lambda: "loaded-source")
    app = create_app()
    monkeypatch.setattr("ml_opt.web.app.source_revision", lambda: "updated-files-on-disk")
    with TestClient(app) as client:
        assert client.get("/api/health").json()["server_revision"] == "loaded-source"


def test_identical_code_in_different_worktrees_does_not_share_a_server(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        (root / "app.py").write_text("same source, separate dataset")
    assert source_revision(first) != source_revision(second)


@pytest.mark.parametrize("health", [{"status": "ready"}, {"status": "ready", "server_revision": "old"}])
def test_old_ready_server_is_skipped_without_stopping_it(monkeypatch, health):
    monkeypatch.setattr(launcher, "source_revision", lambda: "new")
    monkeypatch.setattr(launcher, "_inspect_server", lambda port: health if port == 8000 else None)
    monkeypatch.setattr(launcher, "_port_available", lambda port: port == 8001)
    assert launcher._browser_port(8000) == (8001, False)


def test_second_click_reuses_current_server_on_fallback_port(monkeypatch):
    monkeypatch.setattr(launcher, "source_revision", lambda: "new")
    monkeypatch.setattr(launcher, "_inspect_server", lambda port:
        {"status": "ready", "server_revision": "old" if port == 8000 else "new"})
    assert launcher._browser_port(8000) == (8001, True)


def test_browser_launcher_opens_matching_server_without_starting_another(monkeypatch):
    opened = []
    monkeypatch.setattr(launcher, "_browser_port", lambda _: (8001, True))
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    launcher.main(["--open-browser"])
    assert opened == ["http://127.0.0.1:8001/"]


def test_unrelated_listener_and_exhausted_port_range(monkeypatch):
    monkeypatch.setattr(launcher, "_inspect_server", lambda _: None)
    monkeypatch.setattr(launcher, "_port_available", lambda port: port == 65535)
    assert launcher._browser_port(65534) == (65535, False)
    monkeypatch.setattr(launcher, "_port_available", lambda _: False)
    with pytest.raises(RuntimeError, match="No available port"):
        launcher._browser_port(65534)
