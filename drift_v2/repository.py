from __future__ import annotations

from pathlib import Path

from drift_v2.store import PredictionEventStore, StoredPredictionEvent
from drift_v2.contracts import DemoEvent


class DemoEventRepository:
    """Read immutable demo events from the existing SQLite event store."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._store = PredictionEventStore(self.db_path)

    def load_all_events(
        self,
        limit: int = 100_000,
        source_mode: str | None = None,
    ) -> list[DemoEvent]:
        stored_events = self._load_stored_events(limit=limit, source_mode=source_mode)
        return [
            DemoEvent(
                event_index=index,
                event_id=event.event_id,
                created_at=event.created_at,
                source_mode=event.source_mode,
                raw_text=event.raw_text,
                tokens_stemmed=event.tokens_stemmed,
                num_of_characters=event.num_of_characters,
                num_of_sentences=event.num_of_sentences,
                model_prediction=event.model_prediction,
                model_confidence=event.model_confidence,
                expert_label=event.expert_label,
                expert_confidence=event.expert_confidence,
                hidden_phase=event.hidden_phase,
                hidden_target_label=event.hidden_target_label,
            )
            for index, event in enumerate(stored_events, start=1)
        ]

    def load_source_events(
        self,
        source_mode: str,
        limit: int = 100_000,
    ) -> list[DemoEvent]:
        return self.load_all_events(limit=limit, source_mode=source_mode)

    def load_unlabeled_source_events_for_expert(
        self,
        source_mode: str,
        limit: int,
        since: str | None = None,
    ) -> list[StoredPredictionEvent]:
        return self._store.load_ordered_unlabeled_events_for_expert(
            limit=limit,
            since=since,
            source_mode=source_mode,
        )

    def count_events(self, source_mode: str | None = None) -> int:
        if source_mode is None:
            return self._store.count_events()
        return self._store.count_events_by_source_mode(source_mode)

    def _load_stored_events(
        self,
        limit: int,
        source_mode: str | None,
    ) -> list[StoredPredictionEvent]:
        if source_mode is None:
            return list(reversed(self._store.load_recent_events(limit=limit)))
        return self._store.load_events_by_source_mode(source_mode=source_mode, limit=limit)
