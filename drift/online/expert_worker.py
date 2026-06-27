from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click

from drift.replay.worker import DEFAULT_DB_PATH
from drift.runner.clients import SyntheticApiClient
from drift.runner.config import load_runner_config
from drift.store import PredictionEventStore


@click.command("run")
@click.option(
    "--db-path",
    type=click.Path(exists=True),
    default=str(DEFAULT_DB_PATH),
    show_default=True,
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default="drift/configs/runner.yaml",
    show_default=True,
)
@click.option("--mode", default="debug", show_default=True)
@click.option("--batch-size", type=int, default=10, show_default=True)
@click.option("--lookback-minutes", type=int, default=30, show_default=True)
def main(
    db_path: str,
    config_path: str,
    mode: str,
    batch_size: int,
    lookback_minutes: int,
):
    config = load_runner_config(config_path, mode)
    result = run_expert_batch(
        db_path=Path(db_path),
        synthetic_client=SyntheticApiClient(
            config.synthetic_api_url,
            timeout_seconds=config.request_timeout_seconds,
        ),
        allowed_labels=config.expert_allowed_labels,
        batch_size=batch_size,
        lookback_minutes=lookback_minutes,
    )
    click.echo(json.dumps(result, ensure_ascii=False, indent=2))


def run_expert_batch(
    db_path: Path,
    synthetic_client,
    allowed_labels: list[str],
    batch_size: int = 10,
    lookback_minutes: int = 30,
) -> dict[str, int]:
    store = PredictionEventStore(db_path)
    since = (
        datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
    ).isoformat()
    events = store.load_unlabeled_events_for_expert(limit=batch_size, since=since)
    updated = 0
    failed = 0

    for event in events:
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
        except Exception:
            failed += 1

    return {
        "selected": len(events),
        "updated": updated,
        "failed": failed,
        "total_events": store.count_events(),
    }


if __name__ == "__main__":
    main()
