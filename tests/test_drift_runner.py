from pathlib import Path

import pandas as pd

from drift.runner.cli import run_drift_monitoring
from drift.runner.config import load_runner_config
from drift.runner.metrics import (
    classify_window_status,
    compute_thresholds,
    compute_window_metrics,
)
from drift.runner.pipeline import build_preprocessed_record, build_windows
from drift.runner.prom_metrics import build_prom_metrics_latest
from drift.runner.reference import ensure_reference_snapshot
from drift.runner.scenario import build_phase_sequence, choose_target_label


class FakeSyntheticClient:
    def __init__(self):
        self.generator_calls = 0

    def generate(self, payload):
        self.generator_calls += 1
        label = payload["target_label"]
        texts = {
            "Anxiety": "I feel anxious and restless at night.",
            "Stress": "Work pressure is making me tense and tired.",
            "Depression": "I feel empty and do not enjoy anything.",
            "Normal": "I feel okay and my day was normal.",
            "Suicidal": "I think life is pointless and I want to disappear.",
            "Bipolar": "My mood swings hard and I barely sleep.",
            "Personality disorder": "My relationships are unstable and intense.",
        }
        return {
            "role": "generator",
            "text": texts[label],
            "target_label": label,
            "phase": payload["phase"],
        }

    def label(self, payload):
        text = payload["text"].lower()
        mapping = {
            "anxious": "Anxiety",
            "work pressure": "Stress",
            "empty": "Depression",
            "okay": "Normal",
            "pointless": "Suicidal",
            "mood swings": "Bipolar",
            "unstable": "Personality disorder",
        }
        for keyword, label in mapping.items():
            if keyword in text:
                return {
                    "role": "expert",
                    "label": label,
                    "confidence": 0.9,
                    "reason": keyword,
                }
        return {
            "role": "expert",
            "label": payload["allowed_labels"][0],
            "confidence": 0.5,
            "reason": "fallback",
        }


class FakePredictClient:
    def predict(self, text):
        text = text.lower()
        if "anxious" in text:
            label = "Anxiety"
        elif "work pressure" in text:
            label = "Stress"
        elif "empty" in text:
            label = "Depression"
        else:
            label = "Normal"
        return {
            "prediction": label,
            "confidence": 0.8,
            "probabilities": {label: 0.8},
        }


def test_load_runner_config_debug_and_full():
    debug = load_runner_config("drift/configs/runner.yaml", "debug")
    full = load_runner_config("drift/configs/runner.yaml", "full")

    assert debug.profile.window_size == 20
    assert debug.profile.exploratory_thresholds is True
    assert full.profile.window_size == 100
    assert full.profile.exploratory_thresholds is False


def test_build_phase_sequence():
    sequence = build_phase_sequence({"A_baseline": 2, "B_lexical": 1})
    assert sequence == ["A_baseline", "A_baseline", "B_lexical"]


def test_choose_target_label_prefers_distribution():
    config = load_runner_config("drift/configs/runner.yaml", "debug")
    label = choose_target_label(
        "C_target_shift",
        __import__("random").Random(1),
        {"Anxiety": 1.0},
        config,
    )
    assert label in config.target_shift_distribution


def test_build_preprocessed_record():
    record = build_preprocessed_record("I feel anxious. I cannot sleep.")
    assert record["statement"] == "I feel anxious. I cannot sleep."
    assert isinstance(record["tokens_stemmed"], str)
    assert record["num_of_characters"] > 0
    assert record["num_of_sentences"] == 2


def test_reference_snapshot_build_and_reuse(tmp_path):
    data_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "statement": [
                "I feel anxious.",
                "Work pressure is high.",
                "I feel empty.",
                "I am okay today.",
                "Life feels pointless.",
            ],
            "tokens_stemmed": [
                "i feel anxiou",
                "work pressur is high",
                "i feel empti",
                "i am okay today",
                "life feel pointless",
            ],
            "status": [
                "Anxiety",
                "Stress",
                "Depression",
                "Normal",
                "Suicidal",
            ],
            "num_of_characters": [15, 22, 14, 15, 21],
            "num_of_sentences": [1, 1, 1, 1, 1],
        }
    ).to_csv(data_path)

    config = load_runner_config("drift/configs/runner.yaml", "debug")
    config = config.__class__(**{**config.__dict__, "data_path": data_path, "reference_dir": tmp_path / "reference"})
    df, stats = ensure_reference_snapshot(config)
    assert not df.empty
    assert "class_distribution" in stats
    df2, stats2 = ensure_reference_snapshot(config)
    assert len(df2) == len(df)
    assert stats2["class_distribution"] == stats["class_distribution"]


def test_compute_window_metrics_and_status():
    window_df = pd.DataFrame(
        {
            "tokens_stemmed": [
                "anxious restless",
                "anxious panic",
                "work pressure stress",
                "work pressure tired",
                "empty hopeless",
                "empty tired",
            ],
            "expert_label": [
                "Anxiety",
                "Anxiety",
                "Stress",
                "Stress",
                "Depression",
                "Depression",
            ],
            "model_prediction": [
                "Anxiety",
                "Stress",
                "Stress",
                "Stress",
                "Depression",
                "Depression",
            ],
        }
    )
    reference_stats = {
        "class_distribution": {
            "Anxiety": 0.34,
            "Depression": 0.33,
            "Stress": 0.33,
        },
        "top_tokens": [
            {"token": "anxious", "count": 10},
            {"token": "stress", "count": 10},
            {"token": "empty", "count": 10},
        ],
        "label_token_scores": {
            "Anxiety": [{"token": "anxious", "score": 2.0}],
            "Stress": [{"token": "stress", "score": 2.0}],
            "Depression": [{"token": "empty", "score": 2.0}],
        },
    }
    metrics = compute_window_metrics(
        window_df,
        reference_stats=reference_stats,
        top_k_tokens=3,
        min_examples_per_class=2,
        min_classes=3,
    )
    thresholds = compute_thresholds(
        baseline_windows=[metrics],
        exploratory_thresholds=True,
    )
    status = classify_window_status(metrics, thresholds)
    assert "token_distribution_jsd" in metrics
    assert "target_distribution_jsd" in metrics
    assert "overall_status" in status


def test_build_windows():
    records = [{"sample_id": i, "phase": "A_baseline"} for i in range(50)]
    windows = build_windows(records, window_size=20, step_size=10)
    assert len(windows) == 4
    assert len(windows[0]) == 20


def test_prom_metrics_payload():
    payload = build_prom_metrics_latest(
        run_id="run1",
        mode="debug",
        latest_window={
            "token_distribution_jsd": 0.1,
            "target_distribution_jsd": 0.2,
            "model_expert_disagreement_rate": 0.3,
            "model_expert_macro_f1": 0.7,
            "token_label_association_drift": 0.4,
            "overall_status": "warning",
            "phase": "B_lexical",
            "window_index": 2,
            "token_label_association_status": "ok",
        },
    )
    assert payload["drift_runner_mode"] == "debug"
    assert payload["drift_window_status_code"] == 1


def test_run_drift_monitoring_debug_mode(tmp_path):
    data_path = tmp_path / "data.csv"
    rows = []
    statuses = [
        "Anxiety",
        "Stress",
        "Depression",
        "Normal",
        "Suicidal",
        "Bipolar",
        "Personality disorder",
    ]
    for index in range(80):
        status = statuses[index % len(statuses)]
        rows.append(
            {
                "statement": f"text {status} {index}",
                "tokens_stemmed": f"text {status.lower()}",
                "status": status,
                "num_of_characters": 10 + index,
                "num_of_sentences": 1,
            }
        )
    pd.DataFrame(rows).to_csv(data_path)

    config = load_runner_config("drift/configs/runner.yaml", "debug")
    config = config.__class__(
        **{
            **config.__dict__,
            "data_path": data_path,
            "reference_dir": tmp_path / "reference",
            "output_root": tmp_path / "artifacts",
        }
    )
    summary = run_drift_monitoring(
        config,
        synthetic_client=FakeSyntheticClient(),
        predict_client=FakePredictClient(),
    )
    assert summary["mode"] == "debug"
    assert summary["accepted_samples"] > 0
    run_dir = next((tmp_path / "artifacts").iterdir())
    assert (run_dir / "accepted_samples.jsonl").exists()
    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "prom_metrics_latest.json").exists()
