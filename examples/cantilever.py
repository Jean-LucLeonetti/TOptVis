"""Run from the repository after installing the package in editable mode.

python examples/cantilever.py --optimizer mma
python examples/cantilever.py --optimizer gcmma
"""

from ml_opt.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
