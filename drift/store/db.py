from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


@dataclass(frozen=True)
class PredictionEvent:
    event_id: str
    source_mode: str
    raw_text: str
    tokens_stemmed: str
    num_of_characters: int
    num_of_sentences: int
    model_prediction: str
    model_confidence: float | None
    model_probabilities: dict[str, float]
    expert_label: str | None = None
    expert_confidence: float | None = None
    expert_reason: str | None = None
    hidden_phase: str | None = None
    hidden_target_label: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class StoredPredictionEvent:
    id: int
    event_id: str
    created_at: str
    source_mode: str
    raw_text: str
    tokens_stemmed: str
    num_of_characters: int
    num_of_sentences: int
    model_prediction: str
    model_confidence: float | None
    model_probabilities_json: str
    expert_label: str | None
    expert_confidence: float | None
    expert_reason: str | None
    expert_labeled_at: str | None
    hidden_phase: str | None
    hidden_target_label: str | None
    is_selected_for_expert: bool
    is_training_candidate: bool
    metadata_json: str

    def to_metrics_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "statement": self.raw_text,
            "tokens_stemmed": self.tokens_stemmed,
            "num_of_characters": self.num_of_characters,
            "num_of_sentences": self.num_of_sentences,
            "model_prediction": self.model_prediction,
            "model_confidence": self.model_confidence,
            "expert_label": self.expert_label,
            "expert_confidence": self.expert_confidence,
            "expert_reason": self.expert_reason,
            "source_mode": self.source_mode,
            "created_at": self.created_at,
        }


class PredictionEventStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    def insert_prediction_event(self, event: PredictionEvent) -> bool:
        created_at = event.created_at or _utc_now()
        metadata_json = json.dumps(event.metadata or {}, ensure_ascii=False)
        probabilities_json = json.dumps(event.model_probabilities, ensure_ascii=False)
        expert_labeled_at = _utc_now() if event.expert_label is not None else None

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO prediction_events (
                    event_id,
                    created_at,
                    source_mode,
                    raw_text,
                    tokens_stemmed,
                    num_of_characters,
                    num_of_sentences,
                    model_prediction,
                    model_confidence,
                    model_probabilities_json,
                    expert_label,
                    expert_confidence,
                    expert_reason,
                    expert_labeled_at,
                    hidden_phase,
                    hidden_target_label,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    created_at,
                    event.source_mode,
                    event.raw_text,
                    event.tokens_stemmed,
                    event.num_of_characters,
                    event.num_of_sentences,
                    event.model_prediction,
                    event.model_confidence,
                    probabilities_json,
                    event.expert_label,
                    event.expert_confidence,
                    event.expert_reason,
                    expert_labeled_at,
                    event.hidden_phase,
                    event.hidden_target_label,
                    metadata_json,
                ),
            )
            return cursor.rowcount > 0

    def mark_selected_for_expert(self, event_ids: list[str]) -> int:
        if not event_ids:
            return 0
        placeholders = ",".join("?" for _ in event_ids)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE prediction_events
                SET is_selected_for_expert = 1
                WHERE event_id IN ({placeholders})
                """,
                event_ids,
            )
            return cursor.rowcount

    def update_expert_label(
        self,
        event_id: str,
        expert_label: str,
        expert_confidence: float | None,
        expert_reason: str | None,
        labeled_at: str | None = None,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE prediction_events
                SET
                    expert_label = ?,
                    expert_confidence = ?,
                    expert_reason = ?,
                    expert_labeled_at = ?,
                    is_selected_for_expert = 1
                WHERE event_id = ?
                """,
                (
                    expert_label,
                    expert_confidence,
                    expert_reason,
                    labeled_at or _utc_now(),
                    event_id,
                ),
            )
            return cursor.rowcount > 0

    def load_recent_events(self, limit: int) -> list[StoredPredictionEvent]:
        return self._load_events(
            """
            SELECT *
            FROM prediction_events
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )

    def load_recent_labeled_events(self, limit: int) -> list[StoredPredictionEvent]:
        return self._load_events(
            """
            SELECT *
            FROM prediction_events
            WHERE expert_label IS NOT NULL
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )

    def load_unlabeled_events_for_expert(
        self,
        limit: int,
        since: str | None = None,
    ) -> list[StoredPredictionEvent]:
        if since is None:
            return self._load_events(
                """
                SELECT *
                FROM prediction_events
                WHERE expert_label IS NULL
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (limit,),
            )
        return self._load_events(
            """
            SELECT *
            FROM prediction_events
            WHERE expert_label IS NULL AND created_at >= ?
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (since, limit),
        )

    def load_training_candidates(self) -> list[StoredPredictionEvent]:
        return self._load_events(
            """
            SELECT *
            FROM prediction_events
            WHERE is_training_candidate = 1 AND expert_label IS NOT NULL
            ORDER BY created_at ASC, id ASC
            """,
            (),
        )

    def mark_training_candidates(self, event_ids: list[str]) -> int:
        if not event_ids:
            return 0
        placeholders = ",".join("?" for _ in event_ids)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE prediction_events
                SET is_training_candidate = 1
                WHERE event_id IN ({placeholders}) AND expert_label IS NOT NULL
                """,
                event_ids,
            )
            return cursor.rowcount

    def count_events(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM prediction_events").fetchone()
            return int(row["count"])

    def clear_events(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM prediction_events")
            return cursor.rowcount

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _load_events(
        self,
        query: str,
        params: tuple[Any, ...],
    ) -> list[StoredPredictionEvent]:
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_row_to_event(row) for row in rows]


def _row_to_event(row: sqlite3.Row) -> StoredPredictionEvent:
    data = dict(row)
    data["is_selected_for_expert"] = bool(data["is_selected_for_expert"])
    data["is_training_candidate"] = bool(data["is_training_candidate"])
    return StoredPredictionEvent(**data)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
