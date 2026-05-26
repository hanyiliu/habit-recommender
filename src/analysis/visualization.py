"""
Plotting helpers for Phase 6 analysis of the habit recommender.

All functions:
  - use matplotlib only (no seaborn).
  - return (fig, ax) so the caller can further customize from a notebook.
  - optionally save to disk if `save_path` is provided.
  - accept generic arrays and label lists — no hard-coded project paths.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, save_path: Optional[str]) -> None:
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)


def plot_template_heatmap(
    templates,
    activity_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    cmap: str = "viridis",
):
    """
    Visualize cluster templates as daily schedule heatmaps.

    Parameters
    ----------
    templates : array-like
        Shape (n_clusters, n_time_slots) or (n_time_slots,). Each value is an
        integer activity id.
    activity_labels : sequence of str, optional
        Names for each activity id; used as colorbar tick labels.
    title, save_path, cmap : see module docstring.
    """
    templates = np.atleast_2d(np.asarray(templates))
    if templates.ndim != 2:
        raise ValueError(f"templates must be 1-D or 2-D, got shape {templates.shape}.")
    if templates.size == 0:
        raise ValueError("Empty templates array.")
    n_clusters, n_slots = templates.shape

    fig, ax = plt.subplots(
        figsize=(min(0.25 * n_slots + 2, 16), max(0.4 * n_clusters + 1.5, 3))
    )
    n_classes = int(templates.max() + 1)
    im = ax.imshow(
        templates, aspect="auto", cmap=cmap,
        vmin=0, vmax=max(n_classes - 1, 1),
    )

    ax.set_xlabel("Time slot")
    ax.set_ylabel("Cluster")
    ax.set_yticks(np.arange(n_clusters))
    ax.set_yticklabels([f"C{i}" for i in range(n_clusters)])
    if title:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Activity")
    if activity_labels is not None:
        ticks = np.arange(len(activity_labels))
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(list(activity_labels))
        cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_score_distribution(
    scores,
    cluster_ids=None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    bins: int = 30,
):
    """
    Histogram of composite scores. If `cluster_ids` is provided, overlays one
    semi-transparent histogram per cluster.
    """
    scores = np.asarray(scores).astype(float)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}.")
    if scores.size == 0:
        raise ValueError("Empty scores array.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if cluster_ids is None:
        ax.hist(scores, bins=bins, edgecolor="white", alpha=0.85)
    else:
        cluster_ids = np.asarray(cluster_ids)
        if cluster_ids.shape != scores.shape:
            raise ValueError(
                f"cluster_ids shape {cluster_ids.shape} must match scores {scores.shape}."
            )
        for cid in np.unique(cluster_ids):
            ax.hist(
                scores[cluster_ids == cid],
                bins=bins, alpha=0.45, label=f"Cluster {cid}",
            )
        ax.legend(fontsize=8)

    ax.set_xlabel("Composite score")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_error_by_activity(
    y_true,
    y_pred,
    activity_labels: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = "Error rate by activity (1 - recall)",
):
    """
    Per-class error rate (1 - recall). Highlights the hardest activities to
    predict. Classes present in either y_true or y_pred are shown.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}."
        )
    if y_true.size == 0:
        raise ValueError("Empty input arrays.")

    classes = np.unique(np.concatenate([y_true, y_pred]))
    err = []
    for c in classes:
        mask = y_true == c
        err.append(0.0 if mask.sum() == 0 else 1.0 - (y_pred[mask] == c).mean())
    err = np.array(err)

    fig, ax = plt.subplots(figsize=(max(0.4 * len(classes) + 2, 6), 4.5))
    ax.bar(np.arange(len(classes)), err, edgecolor="white")
    ax.set_xticks(np.arange(len(classes)))
    if activity_labels is not None:
        labels = [
            activity_labels[c] if 0 <= c < len(activity_labels) else str(c)
            for c in classes
        ]
    else:
        labels = [str(c) for c in classes]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Error rate")
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_error_by_time_slot(
    time_slots,
    y_true,
    y_pred,
    save_path: Optional[str] = None,
    title: Optional[str] = "Error rate by time slot",
):
    """
    Per-time-slot error rate. Highlights which moments of the day are hardest
    for the model.
    """
    time_slots = np.asarray(time_slots).astype(int)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if not (time_slots.shape == y_true.shape == y_pred.shape):
        raise ValueError(
            f"Shape mismatch: time_slots {time_slots.shape}, "
            f"y_true {y_true.shape}, y_pred {y_pred.shape}."
        )
    if time_slots.size == 0:
        raise ValueError("Empty input arrays.")

    slots = np.unique(time_slots)
    err = np.array([
        1.0 - (y_true[time_slots == s] == y_pred[time_slots == s]).mean()
        for s in slots
    ])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(slots, err, marker="o", linewidth=1.5)
    ax.set_xlabel("Time slot")
    ax.set_ylabel("Error rate")
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_model_comparison(
    results_dict: Mapping[str, Mapping[str, float]],
    metric_name: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Bar chart comparing one metric across model variants.

    Parameters
    ----------
    results_dict : mapping of model_name -> mapping of metric_name -> float
        e.g. {'GRU4Rec_lambda=0':   {'ndcg@5': 0.41, ...},
              'GRU4Rec_lambda=0.5': {'ndcg@5': 0.46, ...}}
    metric_name : str
        The key from each inner dict to compare.
    """
    if not results_dict:
        raise ValueError("results_dict is empty.")
    models = list(results_dict.keys())
    try:
        values = [float(results_dict[m][metric_name]) for m in models]
    except KeyError as e:
        raise ValueError(f"Missing '{metric_name}' for model {e}.") from e

    fig, ax = plt.subplots(figsize=(max(0.7 * len(models) + 2, 5), 4.5))
    ax.bar(np.arange(len(models)), values, edgecolor="white")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"Model comparison: {metric_name}")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax