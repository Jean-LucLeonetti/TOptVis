"""Launch the local visual interface: python -m ml_opt.web."""

import argparse
import json
import socket
import threading
import urllib.error
import urllib.request
import webbrowser

from .runtime import source_revision


def _inspect_server(port: int) -> dict | None:
    url = f"http://127.0.0.1:{port}/"
    try:
        with urllib.request.urlopen(url + "api/health", timeout=1) as response:
            health = json.load(response)
        with urllib.request.urlopen(url, timeout=1) as response:
            existing_app = b"TOptVis" in response.read()
        if isinstance(health, dict) and existing_app and health.get("status") == "ready":
            return health
    except (OSError, ValueError, urllib.error.URLError):
        pass
    return None


def _port_available(port: int) -> bool:
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _browser_port(port: int) -> tuple[int, bool]:
    """Reuse matching source only; leave older or busy processes undisturbed."""
    revision = source_revision()
    for candidate in range(port, min(port + 20, 65536)):
        health = _inspect_server(candidate)
        if health and health.get("server_revision") == revision:
            return candidate, True
        if health:
            print(f"An older TOptVis server is still on port {candidate}; looking for an updated server port.", flush=True)
        elif _port_available(candidate):
            return candidate, False
    raise RuntimeError("No available port found. Close an unused local server and try again.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Open the local TOptVis visual workspace")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open-browser", action="store_true", help="Open the interface when the server is ready")
    options = parser.parse_args(argv)
    if not 1 <= options.port <= 65535:
        parser.error("port must lie in 1..65535")
    if options.open_browser:
        try:
            options.port, reuse = _browser_port(options.port)
        except RuntimeError as error:
            parser.exit(1, f"error: {error}\n")
        if reuse:
            webbrowser.open(f"http://127.0.0.1:{options.port}/")
            return
    url = f"http://127.0.0.1:{options.port}/"
    print(f"TOptVis: {url}", flush=True)
    import uvicorn
    server = uvicorn.Server(uvicorn.Config("ml_opt.web.app:app", host="127.0.0.1",
                                           port=options.port, log_level="info"))
    finished = threading.Event()
    if options.open_browser:
        def open_when_ready():
            while not finished.wait(0.1):
                if server.started:
                    webbrowser.open(url)
                    return
        threading.Thread(target=open_when_ready, daemon=True).start()
    try:
        server.run()
    finally:
        finished.set()


if __name__ == "__main__":
    main()
