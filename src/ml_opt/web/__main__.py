"""Launch the local visual interface: python -m ml_opt.web."""

import argparse
import threading
import urllib.error
import urllib.request
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the local TOptVis visual workspace")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open-browser", action="store_true", help="Open the interface when the server is ready")
    options = parser.parse_args()
    if not 1 <= options.port <= 65535:
        parser.error("port must lie in 1..65535")
    url = f"http://127.0.0.1:{options.port}/"
    if options.open_browser:
        try:
            with urllib.request.urlopen(url + "api/health", timeout=1) as response:
                import json
                ready = json.load(response).get("status") == "ready"
            with urllib.request.urlopen(url, timeout=1) as response:
                existing_app = b"TOptVis" in response.read()
            if ready and existing_app:
                webbrowser.open(url)
                return
        except (OSError, ValueError, urllib.error.URLError):
            pass
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
