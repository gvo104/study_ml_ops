from pathlib import Path

import pandas as pd

from drift_v2.runner.config import load_runner_config
from drift_v2.store import PredictionEvent, PredictionEventStore
from drift_v2 import DemoEventRepository, build_prefix_windows, build_window_plan
from drift_v2.cli import run_online_loop
from drift_v2.config import DriftV2Config
from drift_v2.pipeline import initialize_offline_replay_artifacts, run_offline_replay_tick, run_online_tick
from drift_v2.playback import advance_playback_state, initial_playback_state


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


def test_demo_event_repository_loads_ordered_events(tmp_path):
    db_path = tmp_path / "events.sqlite"
    store = PredictionEventStore(db_path)
    store.insert_prediction_event(
        PredictionEvent(
            event_id="event-1",
            source_mode="phase_llm",
            raw_text="first",
            tokens_stemmed="first",
            num_of_characters=5,
            num_of_sentences=1,
            model_prediction="Stress",
            model_confidence=0.8,
            model_probabilities={"Stress": 0.8},
            expert_label="Stress",
            expert_confidence=0.9,
            hidden_phase="A_baseline",
            hidden_target_label="Stress",
        )
    )
    store.insert_prediction_event(
        PredictionEvent(
            event_id="event-2",
            source_mode="phase_llm",
            raw_text="second",
            tokens_stemmed="second",
            num_of_characters=6,
            num_of_sentences=1,
            model_prediction="Anxiety",
            model_confidence=0.7,
            model_probabilities={"Anxiety": 0.7},
            expert_label="Anxiety",
            expert_confidence=0.85,
            hidden_phase="B_lexical",
            hidden_target_label="Anxiety",
        )
    )

    events = DemoEventRepository(db_path).load_all_events()

    assert [event.event_index for event in events] == [1, 2]
    assert [event.event_id for event in events] == ["event-1", "event-2"]
    assert events[1].model_prediction == "Anxiety"
    assert events[1].hidden_phase == "B_lexical"


def test_demo_event_repository_filters_source_mode(tmp_path):
    db_path = tmp_path / "events.sqlite"
    store = PredictionEventStore(db_path)
    store.insert_prediction_event(_stored_event("legacy-1", source_mode="online_synthetic"))
    store.insert_prediction_event(_stored_event("v2-1", source_mode="online_synthetic_v2_run"))
    store.insert_prediction_event(_stored_event("v2-2", source_mode="online_synthetic_v2_run"))

    events = DemoEventRepository(db_path).load_source_events("online_synthetic_v2_run")

    assert [event.event_id for event in events] == ["v2-1", "v2-2"]
    assert all(event.source_mode == "online_synthetic_v2_run" for event in events)


def test_window_plan_requires_full_windows():
    events = [
        _event(index)
        for index in range(1, 121)
    ]

    assert build_window_plan(events, visible_events=19, window_size=20, step_size=20).total_complete_windows == 0
    assert build_window_plan(events, visible_events=20, window_size=20, step_size=20).total_complete_windows == 1
    assert build_window_plan(events, visible_events=40, window_size=20, step_size=20).total_complete_windows == 2
    assert build_window_plan(events, visible_events=120, window_size=20, step_size=20).total_complete_windows == 6


def test_build_prefix_windows_returns_complete_batches_only():
    events = [_event(index) for index in range(1, 61)]

    windows = build_prefix_windows(
        events=events,
        visible_events=45,
        window_size=20,
        step_size=20,
    )

    assert len(windows) == 2
    assert windows[0].start_event_index == 1
    assert windows[0].end_event_index == 20
    assert windows[1].start_event_index == 21
    assert windows[1].end_event_index == 40


def test_playback_state_resets_cursor_without_mutating_db():
    state = initial_playback_state(total_events=120, batch_size=20)

    for _ in range(6):
        state = advance_playback_state(state)

    assert state.visible_events == 120
    assert state.cycles_completed == 0

    state = advance_playback_state(state)

    assert state.visible_events == 20
    assert state.cycles_completed == 1


def test_offline_replay_tick_writes_stable_clean_artifacts(tmp_path):
    db_path = tmp_path / "events.sqlite"
    store = PredictionEventStore(db_path)
    for index in range(120):
        label = ["Stress", "Anxiety", "Depression", "Normal"][index % 4]
        phase = (
            "A_baseline"
            if index < 40
            else "B_lexical"
            if index < 60
            else "C_label_distribution"
            if index < 80
            else "D_association"
            if index < 100
            else "A_recovery"
        )
        store.insert_prediction_event(
            PredictionEvent(
                event_id=f"event-{index:03d}",
                source_mode="phase_llm",
                raw_text=f"text {label} {index}",
                tokens_stemmed=f"text {label.lower()}",
                num_of_characters=12 + index,
                num_of_sentences=1,
                model_prediction=label,
                model_confidence=0.75,
                model_probabilities={label: 0.75},
                expert_label=label,
                expert_confidence=0.9,
                hidden_phase=phase,
                hidden_target_label=label,
            )
        )

    config = _build_config(tmp_path, db_path)

    first = run_offline_replay_tick(config=config, visible_events=20)
    second = run_offline_replay_tick(config=config, visible_events=40)

    output_dir = tmp_path / "runs" / "offline_demo_live_v2_debug"
    assert output_dir.exists()
    assert first.complete_window_count == 1
    assert second.complete_window_count == 2
    assert first.run_id == second.run_id == "offline_demo_live_v2"
    assert len((output_dir / "window_metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    recent_events = (output_dir / "recent_events.json").read_text(encoding="utf-8")
    assert '"expert_label": "Stress"' in recent_events
    assert PredictionEventStore(db_path).count_events() == 120


def test_initialize_offline_replay_artifacts_resets_to_empty_snapshot(tmp_path):
    db_path = tmp_path / "events.sqlite"
    store = PredictionEventStore(db_path)
    for index in range(120):
        store.insert_prediction_event(
            PredictionEvent(
                event_id=f"event-{index:03d}",
                source_mode="phase_llm",
                raw_text=f"text {index}",
                tokens_stemmed=f"text {index}",
                num_of_characters=12 + index,
                num_of_sentences=1,
                model_prediction="Stress",
                model_confidence=0.75,
                model_probabilities={"Stress": 0.75},
                expert_label="Stress",
                expert_confidence=0.9,
                hidden_phase="A_baseline",
                hidden_target_label="Stress",
            )
        )

    config = _build_config(tmp_path, db_path)

    bundle = initialize_offline_replay_artifacts(config=config, total_events=120)

    output_dir = tmp_path / "runs" / "offline_demo_live_v2_debug"
    assert bundle.visible_event_count == 0
    assert bundle.complete_window_count == 0
    assert bundle.summary_payload["visible_event_count"] == 0
    assert bundle.summary_payload["complete_window_count"] == 0
    assert bundle.latest_payload["drift_v2_visible_event_count"] == 0
    assert bundle.latest_payload["drift_v2_complete_window_count"] == 0
    assert bundle.latest_payload["labels"]["status"] == "insufficient_data"
    assert bundle.latest_payload["labels"]["pipeline"] == "offline"
    assert (output_dir / "window_metrics.jsonl").read_text(encoding="utf-8") == ""
    assert (output_dir / "recent_events.json").read_text(encoding="utf-8").strip() == '{\n  "events": []\n}'


def test_run_online_tick_uses_clean_source_mode_and_writes_stable_artifacts(tmp_path):
    config = _build_config(
        tmp_path,
        tmp_path / "events.sqlite",
        run_id="online_demo_live_v2",
        pipeline="online",
        source_mode="online_synthetic_v2_online_demo_live_v2",
    )

    bundle, traffic, expert = run_online_tick(
        config=config,
        synthetic_client=FakeSyntheticClient(),
        predict_client=FakePredictClient(),
        hidden_phase="A_baseline",
        traffic_count=20,
        expert_batch_size=20,
        lookback_minutes=1440,
    )
    output_dir = tmp_path / "runs" / "online_demo_live_v2_debug"
    events = DemoEventRepository(config.db_path).load_source_events(config.source_mode)

    assert traffic["inserted"] == 20
    assert expert["updated"] == 20
    assert bundle.visible_event_count == 20
    assert bundle.complete_window_count == 1
    assert bundle.latest_payload["labels"]["pipeline"] == "online"
    assert output_dir.exists()
    assert len(events) == 20
    assert all(event.source_mode == config.source_mode for event in events)


def test_run_online_loop_keeps_single_run_directory(tmp_path):
    config = _build_config(
        tmp_path,
        tmp_path / "events.sqlite",
        run_id="online_demo_live_v2",
        pipeline="online",
        source_mode="online_synthetic_v2_online_demo_live_v2",
        window_size=20,
        step_size=20,
    )

    result = run_online_loop(
        config=config,
        synthetic_client=FakeSyntheticClient(),
        predict_client=FakePredictClient(),
        hidden_phase="A_baseline",
        traffic_count=20,
        expert_batch_size=20,
        lookback_minutes=1440,
        interval_seconds=0,
        max_ticks=2,
    )

    output_dir = tmp_path / "runs" / "online_demo_live_v2_debug"
    assert result["ticks"] == 2
    assert result["inserted"] == 40
    assert result["expert_updated"] == 40
    assert result["visible_events"] == 40
    assert len(list((tmp_path / "runs").iterdir())) == 1
    assert len(output_dir.joinpath("window_metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def _event(index: int):
    from drift_v2.contracts import DemoEvent

    return DemoEvent(
        event_index=index,
        event_id=f"event-{index:03d}",
        created_at="2026-06-29T00:00:00+00:00",
        source_mode="phase_llm",
        raw_text=f"text {index}",
        tokens_stemmed=f"text {index}",
        num_of_characters=10,
        num_of_sentences=1,
        model_prediction="Stress",
        model_confidence=0.8,
        expert_label="Stress",
        expert_confidence=0.9,
        hidden_phase="A_baseline",
        hidden_target_label="Stress",
    )


def _stored_event(event_id: str, source_mode: str) -> PredictionEvent:
    return PredictionEvent(
        event_id=event_id,
        source_mode=source_mode,
        raw_text="text",
        tokens_stemmed="text",
        num_of_characters=10,
        num_of_sentences=1,
        model_prediction="Stress",
        model_confidence=0.8,
        model_probabilities={"Stress": 0.8},
        expert_label="Stress",
        expert_confidence=0.9,
        hidden_phase="A_baseline",
        hidden_target_label="Stress",
    )


def _build_config(
    tmp_path: Path,
    db_path: Path,
    run_id: str = "offline_demo_live_v2",
    pipeline: str = "offline",
    source_mode: str | None = None,
    window_size: int = 20,
    step_size: int = 20,
) -> DriftV2Config:
    runner = load_runner_config("drift_v2/configs/runner.yaml", "debug")
    data_path = _write_reference_data(tmp_path)
    runner = runner.__class__(
        **{
            **runner.__dict__,
            "data_path": data_path,
            "reference_dir": tmp_path / "reference",
            "expert_allowed_labels": ["Stress", "Anxiety", "Depression", "Normal"],
        }
    )
    return DriftV2Config(
        runner=runner,
        db_path=db_path,
        output_root=tmp_path / "runs",
        run_id=run_id,
        pipeline=pipeline,
        source_mode=source_mode,
        event_span=40,
        window_size=window_size,
        step_size=step_size,
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
