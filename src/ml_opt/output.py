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
    fields = {}
    if evaluation.field_densities is not None:
        for name, density in evaluation.field_densities.items():
            fields[f"density_{name}"] = density.reshape(config.ny, config.nx)
            fields[f"displacement_{name}"] = evaluation.field_displacements[name].reshape(-1, 2)
    np.savez_compressed(directory / "result.npz", design=result.design.reshape(config.ny, config.nx),
                        density=evaluation.density.reshape(config.ny, config.nx),
                        displacement=evaluation.displacement.reshape(-1, 2),
                        coordinates=mesh.coordinates, elements=mesh.elements,
                        force=mesh.cantilever_force(config.load, config.load_angle,
                              None if config.load_node_x is None else
                              (config.load_node_x, config.load_node_y)).reshape(-1, 2),
                        state_solver=np.array(config.state_solver), rom_basis_json=np.array(json.dumps(result.rom_basis)), **fields)
    def history_row(entry):
        row = asdict(entry)
        compliances, volumes = row.pop("field_compliances"), row.pop("field_volumes")
        if compliances is not None:
            row.update({f"compliance_{name}": value for name, value in compliances.items()})
            row.update({f"volume_{name}": value for name, value in volumes.items()})
        return row
    with (directory / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history_row(result.history[0])))
        writer.writeheader()
        writer.writerows(history_row(entry) for entry in result.history)
    summary = {
        "config": asdict(config), "stop_reason": result.stop_reason,
        "converged": result.converged, "iterations": result.history[-1].iteration,
        "initial_compliance": result.history[0].compliance,
        "final_compliance": evaluation.compliance,
        "compliance_ratio": evaluation.compliance / result.history[0].compliance,
        "physical_volume_fraction": evaluation.volume_fraction,
        "normalised_volume_constraint": evaluation.constraint,
        "normalised_volume_error": evaluation.volume_error,
        "feasible": evaluation.constraint <= config.feasibility_tolerance,
        "relative_equilibrium_residual": evaluation.residual,
        "state_solves": result.history[-1].state_solves,
        "elapsed_seconds": result.history[-1].elapsed_seconds,
        "state_solver": config.state_solver, "rom_basis": result.rom_basis,
        "objective_model": "approximate frozen Galerkin ROM" if config.state_solver == "rom" else "full-order elasticity",
        "final_projection_beta": evaluation.projection_beta,
    }
    if config.robust:
        epigraph_feasible = evaluation.epigraph_constraint <= config.feasibility_tolerance
        summary.update(field_compliances=evaluation.field_compliances,
                       field_volume_fractions=evaluation.field_volumes,
                       worst_field=evaluation.worst_field,
                       density_field="nominal", displacement_field=evaluation.worst_field,
                       constrained_volume_field="dilated", formulation="three_field_epigraph",
                       epigraph_compliance=evaluation.epigraph_compliance,
                       normalised_epigraph_constraint=evaluation.epigraph_constraint,
                       volume_feasible=summary["feasible"], epigraph_feasible=epigraph_feasible,
                       feasible=summary["feasible"] and epigraph_feasible)
    if config.max_solid_size > 0:
        size_feasible = evaluation.max_size_constraint <= config.feasibility_tolerance
        summary.update(max_size_constraint=evaluation.max_size_constraint,
                       max_size_eroded_volume=evaluation.max_size_eroded_volume,
                       max_size_field="dilated" if config.robust else "nominal",
                       max_size_feasible=size_feasible,
                       feasible=summary["feasible"] and size_feasible)
    if config.max_void_size > 0:
        void_feasible = evaluation.max_void_constraint <= config.feasibility_tolerance
        summary.update(max_void_constraint=evaluation.max_void_constraint,
                       max_void_remaining_volume=evaluation.max_void_remaining_volume,
                       max_void_field="eroded" if config.robust else "nominal",
                       max_void_feasible=void_feasible,
                       feasible=summary["feasible"] and void_feasible)
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
             xlabel="x", ylabel="y", title=f"Cantilever | {config.optimizer.upper()} | {config.state_solver.upper()} | "
             f"{'nominal density | dilated volume' if config.robust else 'volume'} "
             f"{result.evaluation.volume_fraction:.3f} | compliance {result.evaluation.compliance:.3f}")
    axes.set_aspect("equal")
    figure.savefig(directory / "density.png", dpi=160)
    plt.close(figure)

    if config.robust:
        figure, axes = plt.subplots(3, 1, figsize=(10, 9), layout="constrained")
        for axis, (name, density) in zip(axes, result.evaluation.field_densities.items()):
            axis.imshow(density.reshape(config.ny, config.nx), origin="lower",
                        extent=(0, config.length, 0, config.height), cmap="gray_r", vmin=0, vmax=1)
            axis.set(title=f"{name.title()} | volume {result.evaluation.field_volumes[name]:.3f} | "
                     f"compliance {result.evaluation.field_compliances[name]:.3f}", xlabel="x", ylabel="y")
            axis.set_aspect("equal")
        figure.savefig(directory / "robust-fields.png", dpi=160)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    iterations = [entry.iteration for entry in result.history]
    initial = result.history[0].compliance
    axes[0].plot(iterations, [entry.compliance / initial for entry in result.history])
    axes[0].set(xlabel="Iteration", ylabel="Compliance / initial compliance")
    axes[1].plot(iterations, [entry.volume_fraction for entry in result.history],
                 label="Dilated volume" if config.robust else "Physical volume")
    relation = {"lt": "≤", "gt": "≥", "eq": "="}[config.volume_constraint]
    axes[1].axhline(config.volume_fraction, color="tab:red", linestyle="--", label=f"Volume {relation} target")
    axes[1].set(xlabel="Iteration", ylabel="Volume fraction")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(directory / "convergence.png", dpi=160)
    plt.close(figure)
