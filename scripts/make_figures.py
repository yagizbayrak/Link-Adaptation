# Renders the result figures from the evaluation tables into the results folder.

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e3e2df"
SERIES = {"ILLA": "#2a78d6", "OLLA": "#eb6834", "DQN": "#1baf7a"}
GENIE = "#8a8a85"
FIGURES = os.path.join("results", "figures")
TABLES = os.path.join("results", "tables")
ORDER = ["ILLA", "OLLA", "DQN"]


def style(axes):
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=MUTED, labelsize=9)
    axes.yaxis.label.set_color(MUTED)
    axes.xaxis.label.set_color(MUTED)


def new_figure(width, height):
    figure, axes = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    style(axes)
    return figure, axes


def save(figure, name):
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, name), dpi=160, facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {os.path.join(FIGURES, name)}")


def channel_figure(scene):
    data = np.load(os.path.join(TABLES, f"traces_{scene}.npz"))
    effective = data["effective_db"]
    serving = data["serving"]
    metres = np.linspace(0.0, 300.0, effective.size)
    figure, axes = new_figure(10, 3.6)
    axes.plot(metres, effective, color=SERIES["ILLA"], linewidth=0.7, alpha=0.85)
    switches = np.nonzero(np.diff(serving))[0]
    for index, point in enumerate(switches):
        axes.axvline(metres[point], color=MUTED, linewidth=0.8, linestyle=":", alpha=0.7,
                     label="serving cell change" if index == 0 else None)
    axes.set_xlabel("distance along route [m]")
    axes.set_ylabel("effective SINR [dB]")
    axes.set_title("Ray traced effective SINR along the route", color=INK, fontsize=11, loc="left")
    axes.legend(frameon=False, labelcolor=MUTED, fontsize=9, loc="upper left")
    save(figure, f"sinr_trace_{scene}.png")


def bar_figure(scene, frame):
    figure, panels = plt.subplots(1, 3, figsize=(11, 3.8), facecolor=SURFACE)
    metrics = [("goodput_mbps", "goodput [Mb/s]", 1.0, "%.2f"),
               ("bler", "first transmission BLER", 1.0, "%.3f"),
               ("retransmission_rate", "slots spent retransmitting", 100.0, "%.1f%%")]
    for axes, (column, label, scale, form) in zip(panels, metrics):
        style(axes)
        means = [frame[frame.agent == name][column].mean() * scale for name in ORDER]
        errors = [frame[frame.agent == name][column].std() * scale for name in ORDER]
        bars = axes.bar(ORDER, means, yerr=errors, capsize=3, width=0.6,
                        color=[SERIES[name] for name in ORDER], edgecolor=SURFACE, linewidth=2)
        genie = frame[frame.agent == "Genie"][column].mean() * scale
        axes.axhline(genie, color=GENIE, linestyle="--", linewidth=1.6)
        axes.text(2.45, genie, " genie", color=MUTED, fontsize=8, va="bottom", ha="right")
        if column == "bler":
            axes.axhline(0.1, color="#e34948", linestyle=":", linewidth=1.4)
            axes.text(-0.45, 0.1, " target", color="#e34948", fontsize=8, va="bottom")
        for bar, value in zip(bars, means):
            axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), form % value,
                      ha="center", va="bottom", fontsize=9, color=INK)
        axes.set_ylabel(label)
        axes.set_ylim(0, max(max(means) * 1.35, genie * 1.2))
    panels[0].set_title(f"Link adaptation on {scene.replace('_', ' ').title()}", color=INK,
                        fontsize=11, loc="left")
    save(figure, f"comparison_{scene}.png")


def mcs_figure(scene):
    data = np.load(os.path.join(TABLES, f"traces_{scene}.npz"))
    figure, axes = new_figure(10, 3.6)
    window = 400
    for name in ORDER:
        trace = data[name].astype(float)
        smooth = np.convolve(trace, np.ones(window) / window, mode="valid")
        axes.plot(np.linspace(0, 300, smooth.size), smooth, color=SERIES[name],
                  linewidth=1.8, label=name)
    best = data["genie_mcs"].astype(float)
    smooth = np.convolve(best, np.ones(window) / window, mode="valid")
    axes.plot(np.linspace(0, 300, smooth.size), smooth, color=GENIE, linewidth=1.6,
              linestyle="--", label="genie")
    axes.set_xlabel("distance along route [m]")
    axes.set_ylabel("selected MCS index")
    axes.set_title("Chosen modulation and coding scheme along the route", color=INK,
                   fontsize=11, loc="left")
    axes.legend(frameon=False, labelcolor=MUTED, fontsize=9, ncol=4, loc="upper left")
    save(figure, f"mcs_trace_{scene}.png")


def report_figure():
    data = np.load(os.path.join("cache", "report_error.npz"))
    figure, axes = new_figure(7, 3.6)
    axes.plot(data["bins_db"], data["bias_db"], color=SERIES["ILLA"], linewidth=2,
              marker="o", markersize=5, label="mean bias")
    axes.fill_between(data["bins_db"], data["bias_db"] - data["std_db"],
                      data["bias_db"] + data["std_db"], color=SERIES["ILLA"], alpha=0.18,
                      linewidth=0, label="+- one standard deviation")
    axes.axhline(0.0, color=MUTED, linewidth=1.0)
    axes.set_xlabel("true effective SINR [dB]")
    axes.set_ylabel("report error [dB]")
    axes.set_title("Channel quality report error from DMRS least squares estimation",
                   color=INK, fontsize=11, loc="left")
    axes.legend(frameon=False, labelcolor=MUTED, fontsize=9)
    save(figure, "report_error.png")


def training_figure(scene):
    path = os.path.join(TABLES, f"training_{scene}.npy")
    if not os.path.exists(path):
        return
    history = np.load(path)
    figure, axes = new_figure(7, 3.4)
    axes.plot(np.arange(1, history.size + 1), history, color=SERIES["DQN"], linewidth=2,
              marker="o", markersize=5)
    axes.set_xlabel("training episode")
    axes.set_ylabel("goodput [Mb/s]")
    axes.set_title("Deep Q network training progress", color=INK, fontsize=11, loc="left")
    save(figure, f"training_{scene}.png")


def main():
    os.makedirs(FIGURES, exist_ok=True)
    for scene in ("san_francisco", "munich"):
        path = os.path.join(TABLES, f"episodes_{scene}.csv")
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        channel_figure(scene)
        bar_figure(scene, frame)
        mcs_figure(scene)
        training_figure(scene)
    report_figure()


if __name__ == "__main__":
    main()
