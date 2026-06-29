from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from drift_v2.runner.pipeline import build_preprocessed_record
from drift_v2.synthetic_api.schemas import ALLOWED_LABELS
from drift_v2.store import PredictionEvent, PredictionEventStore


PHASE_MAP = {
    "A_baseline": "A",
    "B_lexical": "B",
    "C_label_distribution": "C",
    "D_association": "D",
    "A_recovery": "A",
    "normal": "A",
    "lexical_drift": "B",
    "label_distribution_shift": "C",
    "association_drift": "D",
    "recovery": "A",
}
PHASE_ALIASES = {
    "normal": "A_baseline",
    "lexical_drift": "B_lexical",
    "label_distribution_shift": "C_label_distribution",
    "association_drift": "D_association",
    "recovery": "A_recovery",
}
C_LABEL_SHIFT_DISTRIBUTION = [
    "Stress",
    "Stress",
    "Stress",
    "Stress",
    "Anxiety",
    "Anxiety",
    "Depression",
    "Normal",
]
D_ASSOCIATION_LABELS = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Stress",
]


def normalize_hidden_phase(hidden_phase: str) -> str:
    phase = PHASE_ALIASES.get(hidden_phase, hidden_phase)
    if phase not in PHASE_MAP or phase in PHASE_ALIASES:
        valid = sorted(key for key in PHASE_MAP if key not in PHASE_ALIASES)
        raise ValueError(f"Unknown hidden phase '{hidden_phase}'. Expected one of: {valid}")
    return phase


def choose_online_target_label(
    hidden_phase: str,
    index: int,
    allowed_labels: list[str],
) -> str:
    if not allowed_labels:
        raise ValueError("allowed_labels must not be empty")
    if hidden_phase == "C_label_distribution":
        candidates = [label for label in C_LABEL_SHIFT_DISTRIBUTION if label in allowed_labels]
        if not candidates:
            raise ValueError("C_label_distribution has no labels allowed by config")
        return candidates[index % len(candidates)]
    if hidden_phase == "D_association":
        candidates = [label for label in D_ASSOCIATION_LABELS if label in allowed_labels]
        if not candidates:
            raise ValueError("D_association has no labels allowed by config")
        return candidates[index % len(candidates)]
    return allowed_labels[index % len(allowed_labels)]


def generate_online_events(
    db_path: Path,
    synthetic_client,
    predict_client,
    source_mode: str,
    hidden_phase: str,
    count: int,
    allowed_labels: list[str] | None = None,
    interval_seconds: float = 0.0,
    progress_callback=None,
) -> dict[str, int]:
    store = PredictionEventStore(db_path)
    labels = allowed_labels or ALLOWED_LABELS
    canonical_phase = normalize_hidden_phase(hidden_phase)
    api_phase = PHASE_MAP[canonical_phase]
    inserted = 0
    skipped = 0

    for index in range(count):
        target_label = choose_online_target_label(
            hidden_phase=canonical_phase,
            index=index,
            allowed_labels=labels,
        )
        generated = synthetic_client.generate(
            {
                "role": "generator",
                "phase": api_phase,
                "target_label": target_label,
                "constraints": {"length": "medium", "style": "neutral"},
            }
        )
        text = generated["text"]
        prediction = predict_client.predict(text)
        preprocessed = build_preprocessed_record(text)
        event = PredictionEvent(
            event_id=f"{source_mode}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:12]}",
            source_mode=source_mode,
            raw_text=text,
            tokens_stemmed=preprocessed["tokens_stemmed"],
            num_of_characters=preprocessed["num_of_characters"],
            num_of_sentences=preprocessed["num_of_sentences"],
            model_prediction=prediction["prediction"],
            model_confidence=prediction.get("confidence"),
            model_probabilities=prediction.get("probabilities", {}),
            hidden_phase=canonical_phase,
            hidden_target_label=target_label,
            metadata={
                "generator_phase": api_phase,
                "requested_hidden_phase": hidden_phase,
                "canonical_hidden_phase": canonical_phase,
                "target_label": target_label,
                "model_prediction": prediction["prediction"],
            },
        )
        if store.insert_prediction_event(event):
            inserted += 1
            status = "inserted"
        else:
            skipped += 1
            status = "skipped"
        if progress_callback is not None:
            progress_callback(
                {
                    "index": index + 1,
                    "count": count,
                    "hidden_phase": canonical_phase,
                    "target_label": target_label,
                    "model_prediction": prediction["prediction"],
                    "status": status,
                }
            )
        if interval_seconds > 0:
            time.sleep(interval_seconds)

    return {
        "inserted": inserted,
        "skipped": skipped,
        "total_events": store.count_events_by_source_mode(source_mode),
    }


def label_online_events(
    db_path: Path,
    synthetic_client,
    source_mode: str,
    allowed_labels: list[str],
    batch_size: int,
    lookback_minutes: int,
    progress_callback=None,
) -> dict[str, int]:
    store = PredictionEventStore(db_path)
    since = (
        datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
    ).isoformat()
    events = store.load_ordered_unlabeled_events_for_expert(
        limit=batch_size,
        since=since,
        source_mode=source_mode,
    )
    updated = 0
    failed = 0

    for index, event in enumerate(events, start=1):
        try:
            expert = synthetic_client.label(
                {
                    "role": "expert",
                    "text": event.raw_text,
                    "allowed_labels": allowed_labels,
                }
            )
            if store.update_expert_label(
                event_id=event.event_id,
                expert_label=expert["label"],
                expert_confidence=expert.get("confidence"),
                expert_reason=expert.get("reason"),
            ):
                updated += 1
                status = "updated"
            else:
                status = "skipped"
        except Exception:
            expert = {"label": None, "confidence": None}
            failed += 1
            status = "failed"
        if progress_callback is not None:
            progress_callback(
                {
                    "index": index,
                    "count": len(events),
                    "event_id": event.event_id,
                    "label": expert.get("label"),
                    "confidence": expert.get("confidence"),
                    "status": status,
                }
            )

    return {
        "selected": len(events),
        "updated": updated,
        "failed": failed,
        "total_events": store.count_events_by_source_mode(source_mode),
    }
