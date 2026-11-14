"""Plot the recorded baseline medians and separate diagnostic component times."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/benchmarks/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "benchmarks/results/2026-11-14.json")
    path = parser.parse_args().input
    data = json.loads(path.read_text())
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for method, color in (("mma", "#196964"), ("gcmma", "#df7a40")):
        cases = [item for item in data["core"]
                 if item["case"]["iterations"] == 20 and item["case"]["method"] == method]
        medians = np.array([item["timing"]["median"] for item in cases])
        lower = medians - np.array([item["timing"]["min"] for item in cases])
        upper = np.array([item["timing"]["max"] for item in cases]) - medians
        axes[0].errorbar([item["case"]["nx"] * item["case"]["ny"] for item in cases],
                         medians, yerr=[lower, upper], marker="o", capsize=4,
                         color=color, label=method.upper())
    axes[0].set(xscale="log", xlabel="Elements (log scale)", ylabel="Seconds",
                title="20 updates · uninstrumented core")
    axes[0].set_xticks([300, 1200, 4800], ["300", "1,200", "4,800"])
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    cases = [item for item in data["profiles"] if item["case"]["nx"] in (60, 120)]
    labels = [f"{item['case']['nx']}×{item['case']['ny']}\n{item['case']['method'].upper()}" for item in cases]
    fem, optimizer, backward, other = [], [], [], []
    for item in cases:
        times, wall = item["phase_times"], item["diagnostic_seconds"]
        parts = [times["elasticity_total"],
                 sum(times.get(name, 0) for name in ("mmasub", "gcmmasub", "asymp", "raaupdate", "concheck")),
                 times["torch_grad"]]
        fem.append(100 * parts[0] / wall)
        optimizer.append(100 * parts[1] / wall)
        backward.append(100 * parts[2] / wall)
        other.append(100 * max(0, wall - sum(parts)) / wall)
    bottom = np.zeros(len(cases))
    for values, label, color in ((fem, "Elasticity", "#196964"), (optimizer, "Optimiser", "#df7a40"),
                                 (backward, "Backward", "#5f8ba9"), (other, "Other", "#cad3d8")):
        axes[1].bar(labels, values, bottom=bottom, label=label, color=color, width=.65)
        bottom += np.array(values)
    axes[1].set(ylabel="Share of diagnostic wall time (%)", ylim=(0, 102),
                title="Separate lightweight diagnostic runs")
    axes[1].legend(frameon=False, ncols=4, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.19))
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    date = datetime.fromisoformat(data["timestamp_utc"]).strftime("%d %B %Y").lstrip("0")
    figure.suptitle(f"TOptVis · local CPU performance · {date}", fontsize=13)
    figure.savefig(path.with_name(path.stem + "-performance.png"), dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
