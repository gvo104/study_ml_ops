from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import f1_score

from src.config import TARGET_COLUMN, TEXT_COLUMN


METRIC_DIRECTIONS = {
    "token_distribution_jsd": "higher_is_worse",
    "model_prediction_distribution_jsd": "higher_is_worse",
    "target_distribution_jsd": "higher_is_worse",
    "model_expert_disagreement_rate": "higher_is_worse",
    "model_expert_macro_f1": "lower_is_worse",
    "token_label_association_drift": "higher_is_worse",
}


def compute_window_metrics(
    window_df: pd.DataFrame,
    reference_stats: dict[str, Any],
    top_k_tokens: int,
    min_examples_per_class: int,
    min_classes: int,
) -> dict[str, Any]:
    metrics = compute_pre_label_window_metrics(
        window_df,
        reference_stats=reference_stats,
        top_k_tokens=top_k_tokens,
    )
    metrics.update(
        {
            "target_distribution_jsd": compute_target_distribution_jsd(
                window_df,
                reference_stats["class_distribution"],
            ),
            "expert_confidence_mean": compute_mean_numeric(
                window_df,
                "expert_confidence",
            ),
            "model_expert_disagreement_rate": compute_disagreement_rate(window_df),
            "model_expert_macro_f1": compute_macro_f1(window_df),
        }
    )

    association = compute_token_label_association_drift(
        window_df,
        reference_stats["label_token_scores"],
        min_examples_per_class=min_examples_per_class,
        min_classes=min_classes,
    )
    metrics.update(association)
    return metrics


def compute_pre_label_window_metrics(
    window_df: pd.DataFrame,
    reference_stats: dict[str, Any],
    top_k_tokens: int,
) -> dict[str, Any]:
    return {
        "token_distribution_jsd": compute_token_distribution_jsd(
            window_df,
            reference_stats,
            top_k_tokens=top_k_tokens,
        ),
        "model_prediction_distribution_jsd": compute_model_prediction_distribution_jsd(
            window_df,
            reference_stats["class_distribution"],
        ),
        "model_confidence_mean": compute_mean_numeric(window_df, "model_confidence"),
    }


def compute_token_distribution_jsd(
    df: pd.DataFrame,
    reference_stats: dict[str, Any],
    top_k_tokens: int,
) -> float:
    reference_top_tokens = [
        item["token"] for item in reference_stats["top_tokens"][:top_k_tokens]
    ]
    ref_counts = {item["token"]: item["count"] for item in reference_stats["top_tokens"]}
    cur_counter = Counter()
    for value in df[TEXT_COLUMN]:
        cur_counter.update(_split_tokens(value))

    ref_vector = np.array([ref_counts.get(token, 0) for token in reference_top_tokens], dtype=float)
    cur_vector = np.array([cur_counter.get(token, 0) for token in reference_top_tokens], dtype=float)

    ref_vector = _normalize_counts(ref_vector)
    cur_vector = _normalize_counts(cur_vector)
    return float(jensenshannon(ref_vector, cur_vector, base=2.0) ** 2)


def compute_target_distribution_jsd(
    df: pd.DataFrame,
    reference_distribution: dict[str, float],
) -> float:
    return compute_label_distribution_jsd(
        df,
        label_column="expert_label",
        reference_distribution=reference_distribution,
    )


def compute_model_prediction_distribution_jsd(
    df: pd.DataFrame,
    reference_distribution: dict[str, float],
) -> float:
    return compute_label_distribution_jsd(
        df,
        label_column="model_prediction",
        reference_distribution=reference_distribution,
    )


def compute_label_distribution_jsd(
    df: pd.DataFrame,
    label_column: str,
    reference_distribution: dict[str, float],
) -> float:
    labels = sorted(reference_distribution)
    if df.empty or label_column not in df:
        current_distribution = pd.Series(0.0, index=labels)
    else:
        current_distribution = (
            df[label_column]
            .value_counts(normalize=True)
            .reindex(labels, fill_value=0.0)
        )
        current_distribution = current_distribution.reindex(labels, fill_value=0.0)
    ref_vector = np.array([reference_distribution[label] for label in labels], dtype=float)
    cur_vector = current_distribution.to_numpy(dtype=float)
    return float(jensenshannon(ref_vector, cur_vector, base=2.0) ** 2)


def compute_mean_numeric(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def compute_disagreement_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float((df["model_prediction"] != df["expert_label"]).mean())


def compute_macro_f1(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    labels = sorted(set(df["model_prediction"]).union(set(df["expert_label"])))
    return float(
        f1_score(
            df["expert_label"],
            df["model_prediction"],
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def compute_label_token_scores(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    top_k_per_label: int,
) -> dict[str, list[dict[str, float]]]:
    token_counts_by_label: dict[str, Counter] = defaultdict(Counter)
    total_counts_by_label: dict[str, int] = defaultdict(int)

    for _, row in df.iterrows():
        label = str(row[label_column])
        tokens = _split_tokens(row[text_column])
        token_counts_by_label[label].update(tokens)
        total_counts_by_label[label] += len(tokens)

    vocabulary = sorted(
        {
            token
            for counter in token_counts_by_label.values()
            for token in counter
        }
    )
    vocab_size = max(len(vocabulary), 1)

    results: dict[str, list[dict[str, float]]] = {}
    for label, counter in token_counts_by_label.items():
        rest_counter = Counter()
        rest_total = 0
        for other_label, other_counter in token_counts_by_label.items():
            if other_label == label:
                continue
            rest_counter.update(other_counter)
            rest_total += total_counts_by_label[other_label]

        scored_tokens = []
        for token in vocabulary:
            in_label = counter.get(token, 0)
            in_rest = rest_counter.get(token, 0)
            score = np.log((in_label + 1) / (total_counts_by_label[label] + vocab_size))
            score -= np.log((in_rest + 1) / (rest_total + vocab_size))
            scored_tokens.append({"token": token, "score": float(score)})

        scored_tokens.sort(key=lambda item: item["score"], reverse=True)
        results[label] = scored_tokens[:top_k_per_label]
    return results


def compute_token_label_association_drift(
    df: pd.DataFrame,
    reference_scores: dict[str, list[dict[str, float]]],
    min_examples_per_class: int,
    min_classes: int,
) -> dict[str, Any]:
    window_counts = df["expert_label"].value_counts().to_dict()
    valid_labels = [
        label
        for label, count in window_counts.items()
        if count >= min_examples_per_class and label in reference_scores
    ]
    if len(valid_labels) < min_classes:
        return {
            "token_label_association_drift": None,
            "token_label_association_status": "insufficient_data",
            "token_label_association_valid_labels": valid_labels,
        }

    current_scores = compute_label_token_scores(
        df.rename(columns={"expert_label": TARGET_COLUMN}),
        text_column=TEXT_COLUMN,
        label_column=TARGET_COLUMN,
        top_k_per_label=max(len(tokens) for tokens in reference_scores.values()),
    )
    deltas = []
    for label in valid_labels:
        ref_map = {item["token"]: item["score"] for item in reference_scores[label]}
        cur_map = {item["token"]: item["score"] for item in current_scores.get(label, [])}
        label_deltas = [
            abs(cur_map.get(token, 0.0) - ref_score)
            for token, ref_score in ref_map.items()
        ]
        if label_deltas:
            deltas.append(float(np.mean(label_deltas)))

    if not deltas:
        return {
            "token_label_association_drift": None,
            "token_label_association_status": "insufficient_data",
            "token_label_association_valid_labels": valid_labels,
        }

    return {
        "token_label_association_drift": float(np.mean(deltas)),
        "token_label_association_status": "ok",
        "token_label_association_valid_labels": valid_labels,
    }


def compute_thresholds(
    baseline_windows: list[dict[str, Any]],
    exploratory_thresholds: bool,
) -> dict[str, Any]:
    thresholds: dict[str, Any] = {
        "exploratory_thresholds": exploratory_thresholds,
        "metrics": {},
    }
    for name, direction in METRIC_DIRECTIONS.items():
        values = [
            window[name]
            for window in baseline_windows
            if window.get(name) is not None
        ]
        if not values:
            thresholds["metrics"][name] = {
                "warning": None,
                "critical": None,
                "direction": direction,
            }
            continue
        if direction == "lower_is_worse":
            warning = float(np.percentile(values, 5))
            critical = float(np.percentile(values, 1))
        else:
            warning = float(np.percentile(values, 95))
            critical = float(np.percentile(values, 99))
        thresholds["metrics"][name] = {
            "warning": warning,
            "critical": critical,
            "direction": direction,
        }
    return thresholds


def classify_window_status(
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    statuses: dict[str, str] = {}
    warning_hits = 0
    critical_hits = 0

    for metric_name, threshold in thresholds["metrics"].items():
        value = metrics.get(metric_name)
        if metric_name == "token_label_association_drift" and metrics.get(
            "token_label_association_status"
        ) == "insufficient_data":
            statuses[metric_name] = "insufficient_data"
            continue
        if value is None or threshold["warning"] is None:
            statuses[metric_name] = "unavailable"
            continue

        if threshold["direction"] == "lower_is_worse":
            if value <= threshold["critical"]:
                statuses[metric_name] = "critical"
                critical_hits += 1
            elif value <= threshold["warning"]:
                statuses[metric_name] = "warning"
                warning_hits += 1
            else:
                statuses[metric_name] = "ok"
            continue

        if value >= threshold["critical"]:
            statuses[metric_name] = "critical"
            critical_hits += 1
        elif value >= threshold["warning"]:
            statuses[metric_name] = "warning"
            warning_hits += 1
        else:
            statuses[metric_name] = "ok"

    if critical_hits >= 1 or warning_hits >= 2:
        overall = "critical"
    elif warning_hits >= 1:
        overall = "warning"
    else:
        overall = "ok"
    return {"metric_statuses": statuses, "overall_status": overall}


def _normalize_counts(values: np.ndarray) -> np.ndarray:
    total = values.sum()
    if total <= 0:
        return np.full_like(values, 1.0 / len(values))
    return values / total


def _split_tokens(value: str) -> list[str]:
    return [token for token in str(value).split() if token]
