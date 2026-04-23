"""Visualisation: compare two derivatives from the aggregated summary CSV."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Metric groups — defines layout order in plots
# ---------------------------------------------------------------------------
METRIC_GROUPS: dict[str, list[str]] = {
    "Signal quality": [
        "signal_quality.rms",
        "signal_quality.rms_per_channel_mean",
        "signal_quality.std",
        "signal_quality.kurtosis",
        "line_noise_50hz",
        "snr",
    ],
    "Bad channels": [
        "bad_channels.n_bad",
        "bad_channels.rate",
    ],
    "PSD bands": [
        "psd.delta",
        "psd.theta",
        "psd.alpha",
        "psd.beta",
        "psd.gamma",
    ],
    "ICA": [
        "ica_components.n_components",
        "ica_components.n_excluded",
        "ica_components.exclusion_rate",
    ],
}

_LABELS: dict[str, str] = {
    "signal_quality.rms": "RMS",
    "signal_quality.rms_per_channel_mean": "RMS/ch",
    "signal_quality.std": "Std",
    "signal_quality.kurtosis": "Kurtosis",
    "line_noise_50hz": "Line noise 50Hz",
    "snr": "SNR",
    "bad_channels.n_bad": "N bad ch.",
    "bad_channels.rate": "Bad ch. rate",
    "psd.delta": "PSD δ",
    "psd.theta": "PSD θ",
    "psd.alpha": "PSD α",
    "psd.beta": "PSD β",
    "psd.gamma": "PSD γ",
    "ica_components.n_components": "ICA n comp.",
    "ica_components.n_excluded": "ICA excluded",
    "ica_components.exclusion_rate": "ICA excl. rate",
}

_GROUP_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _load_summary(summary_csv: Path) -> dict[str, dict]:
    data: dict[str, dict] = {}
    with summary_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.pop("derivative")
            data[name] = {k: _parse(v) for k, v in row.items()}
    return data


def _parse(v: str) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _proximity(a: float | None, b: float | None) -> float | None:
    """Relative proximity in [0, 1]. 1 = identical, 0 = maximally different."""
    if a is None or b is None:
        return None
    denom = abs(a) + abs(b)
    if denom < 1e-30:
        return 1.0
    return max(0.0, 1.0 - abs(a - b) / denom)


def compute_proximity_scores(data_a: dict, data_b: dict) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict] = {}
    for group, metrics in METRIC_GROUPS.items():
        result[group] = {}
        for m in metrics:
            result[group][m] = _proximity(data_a.get(m), data_b.get(m))
    return result


def global_proximity(scores: dict[str, dict]) -> float | None:
    all_scores = [v for group in scores.values() for v in group.values() if v is not None]
    return sum(all_scores) / len(all_scores) if all_scores else None

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(
    name_a: str,
    name_b: str,
    scores: dict[str, dict],
    data_a: dict,
    data_b: dict,
    out_path: Path | None = None,
) -> plt.Figure:
    all_metrics = [m for metrics in METRIC_GROUPS.values() for m in metrics]
    all_proximities = [scores[g][m] for g, metrics in METRIC_GROUPS.items() for m in metrics]
    labels = [_LABELS.get(m, m) for m in all_metrics]
    group_sizes = [len(v) for v in METRIC_GROUPS.values()]
    group_names = list(METRIC_GROUPS.keys())
    group_colors = [_GROUP_COLORS[i % len(_GROUP_COLORS)] for i, s in enumerate(group_sizes) for _ in range(s)]

    n = len(all_metrics)
    gp = global_proximity(scores)
    title = (
        f"Derivative comparison:  {name_a}  vs  {name_b}\n"
        f"Global proximity score: {gp:.3f}" if gp is not None
        else f"Derivative comparison:  {name_a}  vs  {name_b}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, max(7, n * 0.48)))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    y = np.arange(n)
    prox_values = [p if p is not None else 0.0 for p in all_proximities]
    is_none = [p is None for p in all_proximities]

    # --- Panel 1: proximity bar chart ---
    bars = ax1.barh(y, prox_values, color=group_colors, alpha=0.85, height=0.6)
    for bar, none, pv in zip(bars, is_none, prox_values):
        if none:
            ax1.text(0.02, bar.get_y() + bar.get_height() / 2, "N/A", va="center", fontsize=8, color="gray")
        else:
            ax1.text(min(pv + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
                     f"{pv:.3f}", va="center", fontsize=8)

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlim(0, 1.15)
    ax1.axvline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Proximity score  (1 = identical)")
    ax1.set_title("Proximity per metric")
    ax1.invert_yaxis()

    cursor = 0
    for i, size in enumerate(group_sizes):
        ax1.axhline(cursor - 0.5, color="black", linewidth=0.5, alpha=0.4)
        ax1.text(1.12, cursor + size / 2 - 0.5, group_names[i],
                 va="center", fontsize=8, color=_GROUP_COLORS[i % len(_GROUP_COLORS)],
                 fontweight="bold", transform=ax1.get_yaxis_transform())
        cursor += size

    # --- Panel 2: raw values (log scale) ---
    def _safe_log(v):
        if v is None or v <= 0:
            return 0.0
        return math.log10(v + 1e-30)

    log_a = [_safe_log(data_a.get(m)) for m in all_metrics]
    log_b = [_safe_log(data_b.get(m)) for m in all_metrics]

    w = 0.28
    ax2.barh(y - w / 2, log_a, height=w, color="#4C72B0", alpha=0.85, label=name_a)
    ax2.barh(y + w / 2, log_b, height=w, color="#DD8452", alpha=0.85, label=name_b)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("log₁₀(value + ε)")
    ax2.set_title("Raw values (log scale)")
    ax2.invert_yaxis()
    ax2.legend(loc="lower right", fontsize=9)

    cursor = 0
    for size in group_sizes:
        ax2.axhline(cursor - 0.5, color="black", linewidth=0.5, alpha=0.4)
        cursor += size

    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compare_derivatives(
    name_a: str,
    name_b: str,
    summary_csv: Path,
    out_dir: Path,
) -> dict:
    """
    Compare two derivatives from the aggregated summary CSV.
    Saves:
      - {name_a}_vs_{name_b}_proximity.json
      - {name_a}_vs_{name_b}.png
    Returns the proximity result dict.
    """
    data = _load_summary(summary_csv)

    if name_a not in data:
        raise ValueError(f"Derivative '{name_a}' not found in {summary_csv}. Available: {list(data)}")
    if name_b not in data:
        raise ValueError(f"Derivative '{name_b}' not found in {summary_csv}. Available: {list(data)}")

    data_a, data_b = data[name_a], data[name_b]
    scores = compute_proximity_scores(data_a, data_b)

    result = {
        "derivative_a": name_a,
        "derivative_b": name_b,
        "global_proximity": global_proximity(scores),
        "n_files_a": data_a.get("n_files"),
        "n_files_b": data_b.get("n_files"),
        "scores": scores,
    }

    slug = f"{name_a}_vs_{name_b}"
    (out_dir / f"{slug}_proximity.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    plot_comparison(name_a, name_b, scores, data_a, data_b, out_path=out_dir / f"{slug}.png")
    plt.close("all")

    return result
