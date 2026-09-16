"""Plotting utilities for Crazyflie Monte Carlo recovery studies.

The plotting layer uses the recovery semantics exported by monte_carlo_core.py:
recovery is distinct from training-envelope violation and hard failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


RECOVERED_OUTCOMES = {
    "recovered_cleanly",
    "recovered_with_training_envelope_violation",
}


def _finish_figure(output_path: Optional[str], show: bool) -> None:
    plt.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def _finite_values(dataframe: pd.DataFrame, column_name: str) -> np.ndarray:
    if column_name not in dataframe.columns:
        return np.array([], dtype=float)
    values = dataframe[column_name].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _column_has_variation(dataframe: pd.DataFrame, column_name: str) -> bool:
    values = _finite_values(dataframe, column_name)
    return values.size > 1 and np.unique(values).size > 1


def compute_empirical_cdf_arrays(data_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    finite = np.asarray(data_array, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    sorted_values = np.sort(finite)
    probabilities = np.arange(1, sorted_values.size + 1, dtype=float) / sorted_values.size
    return sorted_values, probabilities


def generate_recovered_metric_ecdf_plot(
    dataframe: pd.DataFrame,
    metric_column: str,
    metric_label: str,
    output_path: Optional[str] = None,
    quantile_level: float = 0.95,
    show: bool = False,
) -> Optional[float]:
    """ECDF of one performance metric conditional on successful recovery."""
    if "recovered" not in dataframe.columns:
        raise KeyError("Dataframe is missing required column 'recovered'")
    if metric_column not in dataframe.columns:
        raise KeyError(f"Dataframe is missing required column {metric_column!r}")

    recovered = dataframe.loc[dataframe["recovered"].astype(bool)]
    values = _finite_values(recovered, metric_column)
    if values.size == 0:
        return None

    x, probability = compute_empirical_cdf_arrays(values)
    quantile = float(np.quantile(values, quantile_level))

    plt.figure(figsize=(8, 5))
    plt.step(x, probability, where="post", linewidth=2, label="Recovered trials")
    plt.axhline(
        quantile_level,
        linestyle="--",
        linewidth=1.3,
        label=f"{100 * quantile_level:.0f}th percentile level",
    )
    plt.axvline(
        quantile,
        linestyle=":",
        linewidth=1.3,
        label=f"Q{100 * quantile_level:.0f} = {quantile:.4g}",
    )
    plt.xlabel(metric_label)
    plt.ylabel(r"Empirical cumulative probability $\hat{P}(X \leq x\mid recovered)$")
    plt.title(f"{metric_label}: ECDF conditional on recovery")
    plt.grid(alpha=0.3)
    plt.legend()
    _finish_figure(output_path, show)
    return quantile


def render_projected_outcome_map(
    dataframe: pd.DataFrame,
    x_axis_column: str,
    y_axis_column: str,
    color_metric_column: str = "terminal_rmse_position",
    color_metric_label: str = "Terminal position RMSE (m)",
    output_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """2-D projection of high-dimensional Monte Carlo recovery outcomes."""
    required = {"outcome", x_axis_column, y_axis_column, color_metric_column}
    missing = required.difference(dataframe.columns)
    if missing:
        raise KeyError(f"Missing dataframe columns: {sorted(missing)}")

    plt.figure(figsize=(9, 6))
    scatter = None

    recovered_mask = dataframe["outcome"].isin(RECOVERED_OUTCOMES)
    recovered_metric = dataframe.loc[recovered_mask, color_metric_column].to_numpy(dtype=float)
    recovered_metric = recovered_metric[np.isfinite(recovered_metric)]
    norm = None
    if recovered_metric.size:
        vmin = float(np.min(recovered_metric))
        vmax = float(np.max(recovered_metric))
        if np.isclose(vmin, vmax):
            vmax = vmin + max(1e-12, abs(vmin) * 1e-9 + 1e-12)
        norm = Normalize(vmin=vmin, vmax=vmax)

    clean = dataframe["outcome"] == "recovered_cleanly"
    recovered_violation = (
        dataframe["outcome"] == "recovered_with_training_envelope_violation"
    )
    not_recovered = dataframe["outcome"] == "not_recovered_by_horizon"
    hard_failure = dataframe["outcome"] == "hard_failure"
    incomplete = dataframe["outcome"] == "evaluation_guard_reached"

    if clean.any():
        scatter = plt.scatter(
            dataframe.loc[clean, x_axis_column].to_numpy(dtype=float),
            dataframe.loc[clean, y_axis_column].to_numpy(dtype=float),
            c=dataframe.loc[clean, color_metric_column].to_numpy(dtype=float),
            cmap="viridis",
            norm=norm,
            marker="o",
            alpha=0.8,
            edgecolors="k",
            linewidths=0.3,
            label="Recovered cleanly",
        )

    if recovered_violation.any():
        scatter_violation = plt.scatter(
            dataframe.loc[recovered_violation, x_axis_column].to_numpy(dtype=float),
            dataframe.loc[recovered_violation, y_axis_column].to_numpy(dtype=float),
            c=dataframe.loc[recovered_violation, color_metric_column].to_numpy(dtype=float),
            cmap="viridis",
            norm=norm,
            marker="^",
            alpha=0.85,
            edgecolors="k",
            linewidths=0.4,
            label="Recovered; training envelope exceeded",
        )
        if scatter is None:
            scatter = scatter_violation

    if not_recovered.any():
        plt.scatter(
            dataframe.loc[not_recovered, x_axis_column].to_numpy(dtype=float),
            dataframe.loc[not_recovered, y_axis_column].to_numpy(dtype=float),
            marker="X",
            s=75,
            label="Not recovered by horizon",
        )

    if hard_failure.any():
        plt.scatter(
            dataframe.loc[hard_failure, x_axis_column].to_numpy(dtype=float),
            dataframe.loc[hard_failure, y_axis_column].to_numpy(dtype=float),
            marker="s",
            s=65,
            label="Hard failure",
        )

    if incomplete.any():
        plt.scatter(
            dataframe.loc[incomplete, x_axis_column].to_numpy(dtype=float),
            dataframe.loc[incomplete, y_axis_column].to_numpy(dtype=float),
            marker="D",
            s=60,
            label="Evaluation guard reached",
        )

    if scatter is not None:
        color_bar = plt.colorbar(scatter)
        color_bar.set_label(color_metric_label)

    plt.xlabel(x_axis_column.replace("_", " ").title())
    plt.ylabel(y_axis_column.replace("_", " ").title())
    plt.title("Projected Monte Carlo recovery outcomes")
    plt.grid(alpha=0.3)
    plt.legend()
    _finish_figure(output_path, show)


def generate_binned_probability_plot(
    dataframe: pd.DataFrame,
    parameter_column: str,
    event_column: str,
    event_label: str,
    bins: int = 10,
    output_path: Optional[str] = None,
    show: bool = False,
) -> pd.DataFrame:
    """Empirical event probability versus one sampled scalar parameter."""
    required = {parameter_column, event_column}
    missing = required.difference(dataframe.columns)
    if missing:
        raise KeyError(f"Missing dataframe columns: {sorted(missing)}")

    working = dataframe[[parameter_column, event_column]].copy()
    finite = np.isfinite(working[parameter_column].to_numpy(dtype=float))
    working = working.loc[finite]
    if working.empty:
        return pd.DataFrame(
            columns=["parameter_mean", "trials", "event_probability"]
        )

    working["bin"] = pd.cut(
        working[parameter_column], bins=bins, include_lowest=True, duplicates="drop"
    )
    working["event"] = working[event_column].astype(bool)

    grouped = (
        working.groupby("bin", observed=True)
        .agg(
            parameter_mean=(parameter_column, "mean"),
            trials=("event", "size"),
            event_probability=("event", "mean"),
        )
        .reset_index(drop=True)
    )

    x = grouped["parameter_mean"].to_numpy(dtype=float)
    y = grouped["event_probability"].to_numpy(dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.ylim(-0.02, 1.02)
    plt.xlabel(parameter_column.replace("_", " ").title())
    plt.ylabel(f"Empirical {event_label.lower()}")
    plt.title(f"{event_label} vs. {parameter_column.replace('_', ' ')}")
    plt.grid(alpha=0.3)
    _finish_figure(output_path, show)
    return grouped


def generate_outcome_fraction_plot(
    dataframe: pd.DataFrame,
    output_path: Optional[str] = None,
    show: bool = False,
) -> pd.DataFrame:
    """Plot fractions of the explicit Monte Carlo outcome classes."""
    if "outcome" not in dataframe.columns:
        raise KeyError("Dataframe is missing required column 'outcome'")

    order = [
        "recovered_cleanly",
        "recovered_with_training_envelope_violation",
        "not_recovered_by_horizon",
        "hard_failure",
        "evaluation_guard_reached",
    ]
    labels = [
        "Recovered\ncleanly",
        "Recovered +\nenvelope violation",
        "Not recovered\nby horizon",
        "Hard\nfailure",
        "Evaluation\nguard",
    ]

    counts = dataframe["outcome"].value_counts()
    total = max(1, len(dataframe))
    fractions = np.array([float(counts.get(name, 0)) / total for name in order])

    plt.figure(figsize=(9, 5))
    plt.bar(np.arange(len(order)), fractions)
    plt.xticks(np.arange(len(order)), labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Fraction of Monte Carlo trials")
    plt.title("Monte Carlo recovery outcome fractions")
    plt.grid(axis="y", alpha=0.3)
    _finish_figure(output_path, show)

    return pd.DataFrame(
        {
            "outcome": order,
            "count": [int(counts.get(name, 0)) for name in order],
            "fraction": fractions,
        }
    )


def has_parameter_variation(dataframe: pd.DataFrame, column_name: str) -> bool:
    """Public helper used by the study driver to skip degenerate plots."""
    return _column_has_variation(dataframe, column_name)
