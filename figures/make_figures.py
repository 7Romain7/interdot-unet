"""Figures of the README, from the versioned results and the exported U-Net.

    uv run python figures/make_figures.py

Runs on CPU in about a minute. The prediction figure renders a few test scenes with the
official generator (same seeds as the test set) and runs models/unet_lowsnr.pt on them.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
RES = ROOT / "results"

BLUE, ORANGE, GREEN, GREY = "#2a6fdb", "#e8692f", "#1a9e77", "#8c8c8c"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "figure.dpi": 110})

FAMILIES = [  # (name in lofo.json, set prefix, x label, levels)
    ("white", "white_x", "white noise (× σ_pix)", ["0.5", "1", "2"]),
    ("pink", "pink1f_x", "1/f noise (× σ_pix)", ["0.5", "1", "2"]),
    ("drift", "drift_pp", "background drift (peak to peak)", ["1.5", "3", "6"]),
    ("jumps", "jumps_amp", "charge jumps (amplitude)", ["1.5", "3", "6"]),
    ("stripes", "stripes_x", "row stripes (× σ_h)", ["0.5", "1", "2"]),
    ("lowpass", "lowpass_tau", "low-pass τ (px)", ["0.5px", "1px", "2px"]),
    ("saturate", "saturate_c", "saturation c (lower is stronger)", ["10", "5", "2.5"]),
    ("spikes", "spikes_", "outlier pixels (% of pixels)", ["0.1pct", "0.5pct", "2pct"]),
]


def _obj(sets: dict, name: str) -> float:
    v = sets[name]["obj_f1"]
    return v[0] if isinstance(v, list) else v


def fig_predictions() -> None:
    import torch

    from csd import GENERATOR
    from csd.generator import scene_window
    from detection.data_gen import SEED_TEST, _generator_sample
    from detection.evalsets import predict
    from detection.export_dl import load

    model, thr = load(ROOT / "models/unet_lowsnr.pt")
    cfg, win = asdict(GENERATOR), asdict(scene_window(GENERATOR))
    scenes = [3, 12, 21]
    samples = [_generator_sample((SEED_TEST + k, cfg, win)) for k in scenes]
    probs = predict(model, np.stack([s["image"] for s in samples]), torch.device("cpu"))

    cmap = ListedColormap([(0, 0, 0, 0), GREEN, "#d6336c", "#f0a202"])
    fig, axes = plt.subplots(len(scenes), 3, figsize=(9, 3 * len(scenes)))
    for row, (k, s, p) in enumerate(zip(scenes, samples, probs)):
        gt, pred = s["mask"].astype(bool), p > thr
        near_gt = ndimage.binary_dilation(gt, np.ones((3, 3)))
        near_pred = ndimage.binary_dilation(pred, np.ones((3, 3)))
        overlay = np.zeros(gt.shape, int)
        overlay[pred & near_gt] = 1
        overlay[pred & ~near_gt] = 2
        overlay[gt & ~near_pred] = 3
        axes[row, 0].imshow(s["image"], cmap="gray")
        axes[row, 1].imshow(gt, cmap="gray_r")
        axes[row, 2].imshow(s["image"], cmap="gray", alpha=0.6)
        axes[row, 2].imshow(overlay, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        axes[row, 0].set_ylabel(f"test scene {k}\n{len(s['sticks'])} interdots")
        for ax in axes[row]:
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for ax, t in zip(axes[0], ["input CSD", "ground truth", "U-Net prediction"]):
        ax.set_title(t)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREEN, "#d6336c", "#f0a202")]
    fig.legend(handles, ["correct (±1 px)", "false alarm", "missed"], loc="lower center", ncol=3,
               frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "predictions.png")
    plt.close(fig)


def fig_recall_vs_amplitude() -> None:
    summary = json.loads((RES / "robustness/robustness_dl_summary.json").read_text())
    runs = [("fast_unet", "U-Net, uniform intensity", GREY, "--"),
            ("fast_unet_lowsnr", "U-Net, log-uniform intensity", ORANGE, "-"),
            ("fast_transunet_lowsnr", "TransUNet, log-uniform intensity", BLUE, ":")]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for run, label, color, ls in runs:
        rec = summary[run]["sets"]["test"]["recall_by_amplitude"]
        ax.plot(range(len(rec)), list(rec.values()), ls, marker="o", color=color, label=label)
    ax.set_xticks(range(5), ["0–2", "2–4", "4–8", "8–16", "> 16"])
    ax.set_xlabel("interdot amplitude (units of pixel noise σ)")
    ax.set_ylabel("object recall")
    ax.set_ylim(0, 1.02)
    ax.set_title("Test set: recall by interdot amplitude")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "recall_vs_amplitude.png")
    plt.close(fig)


def fig_robustness() -> None:
    plain = json.loads((RES / "robustness/robustness_dl_summary.json").read_text())
    plain = plain["fast_unet_lowsnr"]["sets"]
    per_run = RES / "lofo/per_run"
    none = json.loads((per_run / "lofo_u32_none/eval.json").read_text())["sets"]

    fig, axes = plt.subplots(3, 3, figsize=(11, 8.5))
    for ax, (fam, prefix, xlabel, levels) in zip(axes.flat, FAMILIES):
        held_out = json.loads((per_run / f"lofo_u32_no-{fam}/eval.json").read_text())["sets"]
        names = ["clean"] + [prefix + lv for lv in levels]
        x = range(len(names))
        for sets, label, color, ls in ((plain, "U-Net", GREY, "--"),
                                       (none, "U-Net + PhysInput", BLUE, ":"),
                                       (held_out, "U-Net LOFO (family unseen)", ORANGE, "-")):
            ax.plot(x, [_obj(sets, n) for n in names], ls, marker="o", color=color, label=label)
        ax.set_xticks(list(x), ["0"] + [lv.replace("pct", "").replace("px", "") for lv in levels])
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.02)
    for ax in axes[:, 0]:
        ax.set_ylabel("obj F1")

    ax = axes[2, 2]
    held_out = json.loads((per_run / "lofo_u32_no-polarity/eval.json").read_text())["sets"]
    vals = [_obj(plain, "polarity_flip"), _obj(none, "polarity_flip"),
            _obj(held_out, "polarity_flip")]
    ax.bar(range(3), vals, color=[GREY, BLUE, ORANGE])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    ax.set_xticks(range(3), ["U-Net", "+ PhysInput", "LOFO"])
    ax.set_xlabel("inverted polarity")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="x")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Test obj F1 under each artefact (200 scenes, threshold frozen on val)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "robustness.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_recall_vs_amplitude()
    fig_robustness()
    fig_predictions()
    print(f"figures written to {OUT}")
