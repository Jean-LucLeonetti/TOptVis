"""Serve only the canvas benchmark and the production viewer module on loopback."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resources = {"/": (root / "benchmarks/render.html", "text/html"),
                 "/viewer.js": (root / "src/ml_opt/web/static/viewer.js", "text/javascript")}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in resources:
                self.send_error(404)
                return
            path, content_type = resources[self.path]
            payload = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    server = ThreadingHTTPServer(("127.0.0.1", options.port), Handler)
    print(f"Render benchmark: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
