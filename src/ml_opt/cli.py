"""Command-line interface for the 2D cantilever benchmark."""

import argparse
from pathlib import Path

from .config import CantileverConfig
from .output import save_results
from .runner import HistoryEntry, optimise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SIMP cantilever with PyTorch, PETSc, and MMA/GCMMA")
    defaults = CantileverConfig()
    for name in ("nx", "ny", "max_iterations", "max_inner_iterations", "load_node_x", "load_node_y"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("length", "height", "load", "load_angle", "volume_fraction", "penalisation",
                 "minimum_modulus", "move_limit", "design_tolerance"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--optimizer", choices=("mma", "gcmma"), default=defaults.optimizer)
    parser.add_argument("--state-solver", choices=("fom", "rom"), default="fom",
                        help="Full-order PETSc or a saved frozen Galerkin ROM")
    parser.add_argument("--rom-basis-path", help="Path to a saved basis.npz for --state-solver rom")
    parser.add_argument("--filter-radius", type=float, default=defaults.filter_radius,
                        help="Helmholtz length in model units; 0 disables filtering (default: 0)")
    parser.add_argument("--projection-beta", type=float, default=defaults.projection_beta,
                        help="Heaviside sharpness, fixed during the run; 0 disables projection")
    parser.add_argument("--projection-eta", type=float, default=defaults.projection_eta,
                        help="Heaviside threshold in (0, 1), default: 0.5")
    parser.add_argument("--robust", action="store_true",
                        help="Three-field worst-compliance formulation with dilated-volume constraint")
    parser.add_argument("--robust-eta-offset", type=float, default=defaults.robust_eta_offset,
                        help="Projection threshold offset either side of eta (default: 0.1)")
    parser.add_argument("--field-execution", choices=("sequential", "parallel"),
                        default=defaults.field_execution, help="How robust state solves run")
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
                      f"inner={entry.inner_iterations}  solver={entry.state_solver.upper()}", flush=True)
        result = optimise(config, None if quiet else progress)
        destination = save_results(result, output or Path("outputs") / f"cantilever-{config.optimizer}")
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    if not quiet:
        print(f"Stopped: {result.stop_reason}; results: {destination}")
    return 0
