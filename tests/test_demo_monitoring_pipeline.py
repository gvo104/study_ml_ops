import json
from pathlib import Path

import pandas as pd

from drift.demo.metrics_worker import compute_demo_metrics, compute_demo_thresholds, classify_demo_window_status, export_training_candidates
from drift.demo.phase_worker import generate_phase_demo_data
from drift.replay.worker import run_replay
from drift.runner.config import load_runner_config
from drift.store import PredictionEvent, PredictionEventStore


class FakeSyntheticClient:
    def generate(self, payload):
        label = payload["target_label"]
        phase = payload["phase"]
        return {"text": f"{phase} sample for {label} with pressure anxious empty"}

    def label(self, payload):
        text = payload["text"]
        if "Stress" in text:
            label = "Stress"
        elif "Anxiety" in text:
            label = "Anxiety"
        elif "Depression" in text:
            label = "Depression"
        else:
            label = "Normal"
        return {"label": label, "confidence": 0.93, "reason": "fake expert"}


class FakePredictClient:
    def predict(self, text):
        text_lower = text.lower()
        if "stress" in text_lower or "pressure" in text_lower:
            label = "Stress"
        elif "anxious" in text_lower or "worry" in text_lower:
            label = "Anxiety"
        elif "empty" in text_lower:
            label = "Depression"
        else:
            label = "Normal"
        return {
            "prediction": label,
            "confidence": 0.8,
            "probabilities": {label: 0.8},
        }


def test_phase_worker_stores_hidden_phase_and_writes_replay_cache(tmp_path):
    db_path = tmp_path / "events.sqlite"
    replay_output = tmp_path / "messages.generated.jsonl"

    result = generate_phase_demo_data(
        db_path=db_path,
        synthetic_client=FakeSyntheticClient(),
        predict_client=FakePredictClient(),
        allowed_labels=["Stress", "Anxiety", "Depression", "Normal"],
        phase_counts={"A_baseline": 2, "B_lexical": 1, "C_label_distribution": 1, "D_association": 1},
        interval_seconds=0,
        replay_output=replay_output,
        reset_db=True,
    )

    events = list(reversed(PredictionEventStore(db_path).load_recent_events(limit=10)))
    assert result.inserted == 5
    assert {event.hidden_phase for event in events} == {
        "A_baseline",
        "B_lexical",
        "C_label_distribution",
        "D_association",
    }
    assert all(json.loads(event.metadata_json)["generator_phase"] in {"A", "B", "C", "D"} for event in events)
    assert replay_output.exists()
    assert len(replay_output.read_text(encoding="utf-8").splitlines()) == 5


def test_event_store_insert_update_and_recent_reads(tmp_path):
    store = PredictionEventStore(tmp_path / "events.sqlite")
    event = _event("event-1", expert_label=None)

    assert store.insert_prediction_event(event) is True
    assert store.insert_prediction_event(event) is False
    assert store.count_events() == 1

    assert store.update_expert_label("event-1", "Stress", 0.91, "work pressure")
    recent = store.load_recent_events(limit=10)
    labeled = store.load_recent_labeled_events(limit=10)

    assert len(recent) == 1
    assert len(labeled) == 1
    assert labeled[0].expert_label == "Stress"
    assert labeled[0].is_selected_for_expert is True


def test_replay_worker_is_idempotent(tmp_path):
    dataset = tmp_path / "messages.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_id": "replay-1",
                        "text": "Work pressure makes me tense.",
                        "expert_label": "Stress",
                        "expert_confidence": 0.9,
                        "expert_reason": "stress signal",
                        "hidden_phase": "normal",
                        "hidden_target_label": "Stress",
                    }
                ),
                json.dumps(
                    {
                        "event_id": "replay-2",
                        "text": "I feel anxious at night.",
                        "expert_label": "Anxiety",
                        "expert_confidence": 0.92,
                        "expert_reason": "anxiety signal",
                        "hidden_phase": "lexical_drift",
                        "hidden_target_label": "Anxiety",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "events.sqlite"

    first = run_replay(dataset, db_path, FakePredictClient(), interval_seconds=0)
    second = run_replay(dataset, db_path, FakePredictClient(), interval_seconds=0)

    assert first["inserted"] == 2
    assert second["inserted"] == 0
    assert second["skipped"] == 2
    assert PredictionEventStore(db_path).count_events() == 2


def test_demo_metrics_hide_phase_and_mark_candidates(tmp_path):
    data_path = _write_reference_data(tmp_path)
    db_path = tmp_path / "events.sqlite"
    store = PredictionEventStore(db_path)
    labels = ["Stress", "Anxiety", "Depression", "Normal"]
    for index in range(24):
        label = labels[index % len(labels)]
        prediction = "Normal" if index >= 20 else label
        store.insert_prediction_event(
            _event(
                f"event-{index:03d}",
                expert_label=label,
                model_prediction=prediction,
                hidden_phase="normal" if index < 12 else "association_drift",
            )
        )

    config = load_runner_config("drift/configs/runner.yaml", "debug")
    config = config.__class__(
        **{
            **config.__dict__,
            "data_path": data_path,
            "reference_dir": tmp_path / "reference",
        }
    )

    summary = compute_demo_metrics(
        db_path=db_path,
        config=config,
        output_root=tmp_path / "runs",
        window_size=8,
        step_size=4,
    )
    run_dir = next((tmp_path / "runs").iterdir())
    prom_payload = json.loads((run_dir / "prom_metrics_latest.json").read_text())
    window_lines = (run_dir / "window_metrics.jsonl").read_text().splitlines()

    assert summary["window_count"] > 0
    assert "phase" not in prom_payload["labels"]
    assert '"phase": "association_drift"' not in json.dumps(prom_payload)
    assert any(json.loads(line)["phase"] == "association_drift" for line in window_lines)
    assert summary["training_candidates_marked"] >= 0


def test_demo_thresholds_keep_baseline_ok_and_require_baseline():
    baseline_windows = [
        {"token_distribution_jsd": 0.10, "model_expert_macro_f1": 0.90},
        {"token_distribution_jsd": 0.12, "model_expert_macro_f1": 0.88},
    ]

    thresholds = compute_demo_thresholds(baseline_windows, exploratory_thresholds=True)
    ok_status = classify_demo_window_status(
        {"token_distribution_jsd": 0.12, "model_expert_macro_f1": 0.88},
        thresholds,
    )
    drift_status = classify_demo_window_status(
        {"token_distribution_jsd": 0.20, "model_expert_macro_f1": 0.70},
        thresholds,
    )
    missing_status = classify_demo_window_status(
        {"token_distribution_jsd": 0.20},
        compute_demo_thresholds([], exploratory_thresholds=True),
    )

    assert thresholds["calibration_status"] == "ok"
    assert ok_status["overall_status"] == "ok"
    assert drift_status["overall_status"] == "critical"
    assert missing_status["overall_status"] == "insufficient_data"


def test_export_training_candidates_excludes_hidden_fields_by_default(tmp_path):
    db_path = tmp_path / "events.sqlite"
    store = PredictionEventStore(db_path)
    store.insert_prediction_event(_event("candidate-1", expert_label="Stress"))
    store.mark_training_candidates(["candidate-1"])

    output = tmp_path / "candidates.csv"
    count = export_training_candidates(db_path, output)

    assert count == 1
    content = output.read_text(encoding="utf-8")
    assert "hidden_phase" not in content
    assert "candidate-1" in content


def _event(
    event_id,
    expert_label="Stress",
    model_prediction="Stress",
    hidden_phase="normal",
):
    return PredictionEvent(
        event_id=event_id,
        source_mode="replay",
        raw_text="Work pressure makes me stressed.",
        tokens_stemmed="work pressur make me stress",
        num_of_characters=32,
        num_of_sentences=1,
        model_prediction=model_prediction,
        model_confidence=0.8,
        model_probabilities={model_prediction: 0.8},
        expert_label=expert_label,
        expert_confidence=0.9 if expert_label else None,
        expert_reason="test expert" if expert_label else None,
        hidden_phase=hidden_phase,
        hidden_target_label=expert_label,
    )


def _write_reference_data(tmp_path: Path) -> Path:
    rows = []
    statuses = ["Stress", "Anxiety", "Depression", "Normal"]
    for index in range(40):
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
    data_path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(data_path)
    return data_path
