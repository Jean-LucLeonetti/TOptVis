"""Save numerical results, an iteration history, and headless diagnostic figures."""

import csv
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from .fem.mesh import StructuredMesh
from .runner import OptimizationResult


def save_results(result: OptimizationResult, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config, evaluation = result.config, result.evaluation
    mesh = StructuredMesh(config.nx, config.ny, config.length, config.height)
    np.savez_compressed(directory / "result.npz", design=result.design.reshape(config.ny, config.nx),
                        density=evaluation.density.reshape(config.ny, config.nx),
                        displacement=evaluation.displacement.reshape(-1, 2),
                        coordinates=mesh.coordinates, elements=mesh.elements,
                        force=mesh.cantilever_force(config.load, config.load_angle,
                              None if config.load_node_x is None else
                              (config.load_node_x, config.load_node_y)).reshape(-1, 2))
    with (directory / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(result.history[0])))
        writer.writeheader()
        writer.writerows(asdict(entry) for entry in result.history)
    summary = {
        "config": asdict(config), "stop_reason": result.stop_reason,
        "converged": result.converged, "iterations": result.history[-1].iteration,
        "initial_compliance": result.history[0].compliance,
        "final_compliance": evaluation.compliance,
        "compliance_ratio": evaluation.compliance / result.history[0].compliance,
        "physical_volume_fraction": evaluation.volume_fraction,
        "normalised_volume_constraint": evaluation.constraint,
        "feasible": evaluation.constraint <= config.feasibility_tolerance,
        "relative_equilibrium_residual": evaluation.residual,
        "state_solves": result.history[-1].state_solves,
        "elapsed_seconds": result.history[-1].elapsed_seconds,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _save_figures(result, directory)
    return directory.resolve()


def _save_figures(result: OptimizationResult, directory: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((directory / ".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    config = result.config
    figure, axes = plt.subplots(figsize=(10, 4), layout="constrained")
    axes.imshow(result.evaluation.density.reshape(config.ny, config.nx),
                origin="lower", extent=(0, config.length, 0, config.height),
                cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    pad = 0.035 * config.length
    axes.add_patch(Rectangle((-pad, 0), pad, config.height, facecolor="0.85", hatch="///"))
    anchor = np.array([config.length, config.height / 2]) if config.load_node_x is None else np.array([
        config.load_node_x * config.length / config.nx, config.load_node_y * config.height / config.ny])
    direction = np.array([np.cos(np.deg2rad(config.load_angle)), np.sin(np.deg2rad(config.load_angle))])
    tip = anchor + direction * 0.25 * config.height
    axes.annotate("", xy=tip, xytext=anchor,
                  arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 2})
    axes.set(xlim=(min(-pad, tip[0] - pad), max(config.length + pad, tip[0] + pad)),
             ylim=(min(-0.03 * config.height, tip[1] - pad), max(1.03 * config.height, tip[1] + pad)),
             xlabel="x", ylabel="y", title=f"Cantilever | {config.optimizer.upper()} | "
             f"volume {result.evaluation.volume_fraction:.3f} | compliance {result.evaluation.compliance:.3f}")
    axes.set_aspect("equal")
    figure.savefig(directory / "density.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    iterations = [entry.iteration for entry in result.history]
    initial = result.history[0].compliance
    axes[0].plot(iterations, [entry.compliance / initial for entry in result.history])
    axes[0].set(xlabel="Iteration", ylabel="Compliance / initial compliance")
    axes[1].plot(iterations, [entry.volume_fraction for entry in result.history], label="Physical volume")
    axes[1].axhline(config.volume_fraction, color="tab:red", linestyle="--", label="Volume limit")
    axes[1].set(xlabel="Iteration", ylabel="Volume fraction")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(directory / "convergence.png", dpi=160)
    plt.close(figure)
