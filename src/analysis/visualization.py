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


def plot_ablation_comparison(
    results_dict: Mapping[str, Mapping[str, float]],
    metric_names: Sequence[str],
    save_path: Optional[str] = None,
    title: Optional[str] = "Ablation comparison",
):
    """
    Grouped bar chart comparing several metrics across model variants. Useful
    for the lambda=0 vs lambda>0 (BPR-only vs BPR+KL) ablation: one group per
    variant, one bar per metric.

    Parameters
    ----------
    results_dict : mapping of variant_name -> mapping of metric_name -> float
    metric_names : sequence of metric keys present in every inner dict
    """
    if not results_dict:
        raise ValueError("results_dict is empty.")
    if not metric_names:
        raise ValueError("metric_names is empty.")

    variants = list(results_dict.keys())
    n_v = len(variants)
    n_m = len(metric_names)
    values = np.zeros((n_v, n_m), dtype=float)
    for i, v in enumerate(variants):
        for j, m in enumerate(metric_names):
            if m not in results_dict[v]:
                raise ValueError(f"Metric '{m}' missing for variant '{v}'.")
            values[i, j] = float(results_dict[v][m])

    fig, ax = plt.subplots(figsize=(max(1.1 * n_v + 2, 6), 4.5))
    width = 0.8 / n_m
    x = np.arange(n_v)
    for j, m in enumerate(metric_names):
        offset = (j - (n_m - 1) / 2) * width
        bars = ax.bar(x + offset, values[:, j], width=width, label=m, edgecolor="white")
        for rect, v in zip(bars, values[:, j]):
            ax.text(
                rect.get_x() + rect.get_width() / 2, v,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=min(n_m, 4))
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_alignment_tradeoff(
    results,
    fidelity_key: str = "ndcg@5",
    alignment_key: str = "alignment_ndcg@5",
    lambda_key: str = "lambda",
    selected_lambda: Optional[float] = None,
    floor: Optional[float] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Fidelity-vs-alignment tradeoff curve parameterized by λ.

    Parameters
    ----------
    results : sequence of mappings
        Each item has ``lambda_key``, ``fidelity_key`` (agreement with the
        user's actual next activity, y-axis) and ``alignment_key`` (agreement
        with the routine template, x-axis).
    selected_lambda : float, optional
        The chosen λ*; its point is highlighted.
    floor : float, optional
        Fidelity floor; drawn as a horizontal reference line.

    Returns ``(fig, ax)``.
    """
    if not results:
        raise ValueError("results is empty.")
    pts = sorted(results, key=lambda r: r[lambda_key])
    xs = [float(r[alignment_key]) for r in pts]
    ys = [float(r[fidelity_key]) for r in pts]
    lams = [r[lambda_key] for r in pts]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(xs, ys, "-o", color="#3b6", zorder=2)
    for x, y, lam in zip(xs, ys, lams):
        ax.annotate(f"λ={lam}", (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)

    if selected_lambda is not None:
        for x, y, lam in zip(xs, ys, lams):
            if lam == selected_lambda:
                ax.scatter([x], [y], s=160, facecolors="none",
                           edgecolors="crimson", linewidths=2, zorder=3,
                           label=f"selected λ*={lam}")
                break

    if floor is not None:
        ax.axhline(floor, color="gray", ls="--", lw=1,
                   label=f"fidelity floor = {floor:.3f}")

    ax.set_xlabel(f"Alignment ({alignment_key}) — agreement with routine template")
    ax.set_ylabel(f"Fidelity ({fidelity_key}) — agreement with real behavior")
    ax.set_title(title or "Fidelity vs. alignment tradeoff over λ")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax
