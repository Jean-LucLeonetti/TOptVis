"""Launch the local visual interface: python -m ml_opt.web."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the local TOptVis visual workspace")
    parser.add_argument("--port", type=int, default=8000)
    options = parser.parse_args()
    if not 1 <= options.port <= 65535:
        parser.error("port must lie in 1..65535")
    import uvicorn
    uvicorn.run("ml_opt.web.app:app", host="127.0.0.1", port=options.port, log_level="info")


if __name__ == "__main__":
    main()
