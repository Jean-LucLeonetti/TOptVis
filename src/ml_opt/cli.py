"""Command-line interface for the 2D cantilever benchmark."""

import argparse
from pathlib import Path

from .config import CantileverConfig
from .output import save_results
from .runner import HistoryEntry, optimise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SIMP cantilever with PyTorch, PETSc, and MMA/GCMMA")
    defaults = CantileverConfig()
    for name in ("nx", "ny", "max_iterations", "max_inner_iterations"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("length", "height", "volume_fraction", "penalisation",
                 "minimum_modulus", "move_limit", "design_tolerance"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--optimizer", choices=("mma", "gcmma"), default=defaults.optimizer)
    parser.add_argument("--output", type=Path, help="Output directory (default: outputs/cantilever-<optimizer>)")
    parser.add_argument("--quiet", action="store_true")
    options = vars(parser.parse_args(argv))
    output, quiet = options.pop("output"), options.pop("quiet")
    try:
        config = CantileverConfig(**options)
        def progress(entry: HistoryEntry) -> None:
            if entry.iteration % 5 == 0:
                print(f"{entry.iteration:4d}  compliance={entry.compliance:.6f}  "
                      f"volume={entry.volume_fraction:.5f}  change={entry.design_change:.4f}  "
                      f"inner={entry.inner_iterations}", flush=True)
        result = optimise(config, None if quiet else progress)
        destination = save_results(result, output or Path("outputs") / f"cantilever-{config.optimizer}")
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    if not quiet:
        print(f"Stopped: {result.stop_reason}; results: {destination}")
    return 0
