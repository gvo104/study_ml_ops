from __future__ import annotations

import fcntl
import json
import time
from contextlib import contextmanager
from pathlib import Path

import click

from drift_v2.config import (
    DEFAULT_DB_PATH,
    DEFAULT_EVENT_SPAN,
    DEFAULT_ONLINE_RUN_ID,
    DEFAULT_ONLINE_SOURCE_MODE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RUN_ID,
    DEFAULT_STEP_SIZE,
    DEFAULT_WINDOW_SIZE,
    DriftV2Config,
    load_drift_v2_config,
)
from drift_v2.runner.clients import PredictApiClient, SyntheticApiClient
from drift_v2.pipeline import initialize_artifacts, initialize_offline_replay_artifacts, run_offline_replay_tick, run_online_tick
from drift_v2.playback import advance_playback_state, initial_playback_state
from drift_v2.repository import DemoEventRepository


@click.group()
def main():
    """Clean drift v2 offline demo pipeline."""


@main.command("offline")
@click.option("--db-path", type=click.Path(exists=True), default=str(DEFAULT_DB_PATH), show_default=True)
@click.option("--config", "config_path", type=click.Path(exists=True), default="drift_v2/configs/runner.yaml", show_default=True)
@click.option("--mode", default="debug", show_default=True)
@click.option("--run-id", default=DEFAULT_RUN_ID, show_default=True)
@click.option("--output-root", type=click.Path(), default=str(DEFAULT_OUTPUT_ROOT), show_default=True)
@click.option("--event-span", type=int, default=DEFAULT_EVENT_SPAN, show_default=True)
@click.option("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, show_default=True)
@click.option("--step-size", type=int, default=DEFAULT_STEP_SIZE, show_default=True)
@click.option("--batch-size", type=int, default=DEFAULT_WINDOW_SIZE, show_default=True)
@click.option("--interval-seconds", type=float, default=15.0, show_default=True)
@click.option("--max-ticks", type=int, default=0, show_default=True)
def offline_cmd(
    db_path: str,
    config_path: str,
    mode: str,
    run_id: str,
    output_root: str,
    event_span: int,
    window_size: int,
    step_size: int,
    batch_size: int,
    interval_seconds: float,
    max_ticks: int,
):
    config = load_drift_v2_config(
        config_path=config_path,
        mode=mode,
        db_path=db_path,
        output_root=output_root,
        run_id=run_id,
        pipeline="offline",
        event_span=event_span,
        window_size=window_size,
        step_size=step_size,
    )
    result = run_offline_loop(
        config=config,
        batch_size=batch_size,
        interval_seconds=interval_seconds,
        max_ticks=max_ticks,
    )
    click.echo(json.dumps(result, ensure_ascii=False, indent=2))


@main.command("online")
@click.option("--db-path", type=click.Path(), default=str(DEFAULT_DB_PATH), show_default=True)
@click.option("--config", "config_path", type=click.Path(exists=True), default="drift_v2/configs/runner.yaml", show_default=True)
@click.option("--mode", default="debug", show_default=True)
@click.option("--run-id", default=DEFAULT_ONLINE_RUN_ID, show_default=True)
@click.option("--output-root", type=click.Path(), default=str(DEFAULT_OUTPUT_ROOT), show_default=True)
@click.option("--source-mode", default=DEFAULT_ONLINE_SOURCE_MODE, show_default=True)
@click.option("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, show_default=True)
@click.option("--step-size", type=int, default=DEFAULT_STEP_SIZE, show_default=True)
@click.option("--event-span", type=int, default=DEFAULT_EVENT_SPAN, show_default=True)
@click.option("--traffic-count", type=int, default=DEFAULT_WINDOW_SIZE, show_default=True)
@click.option("--expert-batch-size", type=int, default=DEFAULT_WINDOW_SIZE, show_default=True)
@click.option("--lookback-minutes", type=int, default=1440, show_default=True)
@click.option("--hidden-phase", default="A_baseline", show_default=True)
@click.option("--interval-seconds", type=float, default=15.0, show_default=True)
@click.option("--max-ticks", type=int, default=0, show_default=True)
def online_cmd(
    db_path: str,
    config_path: str,
    mode: str,
    run_id: str,
    output_root: str,
    source_mode: str,
    window_size: int,
    step_size: int,
    event_span: int,
    traffic_count: int,
    expert_batch_size: int,
    lookback_minutes: int,
    hidden_phase: str,
    interval_seconds: float,
    max_ticks: int,
):
    resolved_source_mode = source_mode
    if source_mode == DEFAULT_ONLINE_SOURCE_MODE:
        resolved_source_mode = f"{source_mode}_{run_id}"
    config = load_drift_v2_config(
        config_path=config_path,
        mode=mode,
        db_path=db_path,
        output_root=output_root,
        run_id=run_id,
        pipeline="online",
        source_mode=resolved_source_mode,
        event_span=event_span,
        window_size=window_size,
        step_size=step_size,
    )
    synthetic_client = SyntheticApiClient(
        config.runner.synthetic_api_url,
        timeout_seconds=config.runner.request_timeout_seconds,
    )
    predict_client = PredictApiClient(
        config.runner.predict_api_url,
        timeout_seconds=config.runner.request_timeout_seconds,
    )
    result = run_online_loop(
        config=config,
        synthetic_client=synthetic_client,
        predict_client=predict_client,
        hidden_phase=hidden_phase,
        traffic_count=traffic_count,
        expert_batch_size=expert_batch_size,
        lookback_minutes=lookback_minutes,
        interval_seconds=interval_seconds,
        max_ticks=max_ticks,
    )
    click.echo(json.dumps(result, ensure_ascii=False, indent=2))


def run_offline_loop(
    config: DriftV2Config,
    batch_size: int,
    interval_seconds: float,
    max_ticks: int,
) -> dict[str, object]:
    with _acquire_run_lock(config.output_root, config.run_id, config.runner.mode):
        total_events = len(DemoEventRepository(config.db_path).load_all_events())
        initialize_offline_replay_artifacts(config=config, total_events=total_events)
        state = initial_playback_state(total_events=total_events, batch_size=batch_size)
        tick = 0
        last_bundle = None

        while max_ticks <= 0 or tick < max_ticks:
            tick += 1
            state = advance_playback_state(state)
            last_bundle = run_offline_replay_tick(
                config=config,
                visible_events=state.visible_events,
            )
            click.echo(
                "[drift-v2-offline] "
                f"tick={tick} "
                f"cycle={state.cycles_completed} "
                f"visible_events={state.visible_events} "
                f"windows={last_bundle.complete_window_count} "
                f"status={last_bundle.latest_payload['labels']['status']}"
            )
            if interval_seconds > 0 and (max_ticks <= 0 or tick < max_ticks):
                time.sleep(interval_seconds)

        return {
            "run_id": config.run_id,
            "ticks": tick,
            "cycles_completed": state.cycles_completed,
            "visible_events": state.visible_events,
            "total_events": state.total_events,
            "latest_summary": last_bundle.summary_payload if last_bundle else None,
        }


def run_online_loop(
    config: DriftV2Config,
    synthetic_client,
    predict_client,
    hidden_phase: str,
    traffic_count: int,
    expert_batch_size: int,
    lookback_minutes: int,
    interval_seconds: float,
    max_ticks: int,
) -> dict[str, object]:
    if not config.source_mode:
        raise ValueError("online drift_v2 config requires source_mode")
    with _acquire_run_lock(config.output_root, config.run_id, config.runner.mode):
        repository = DemoEventRepository(config.db_path)
        total_events = repository.count_events(source_mode=config.source_mode)
        initialize_artifacts(config=config, total_events=total_events, visible_events=0)
        tick = 0
        last_bundle = None
        inserted_total = 0
        updated_total = 0

        while max_ticks <= 0 or tick < max_ticks:
            tick += 1
            last_bundle, traffic, expert = run_online_tick(
                config=config,
                synthetic_client=synthetic_client,
                predict_client=predict_client,
                hidden_phase=hidden_phase,
                traffic_count=traffic_count,
                expert_batch_size=expert_batch_size,
                lookback_minutes=lookback_minutes,
            )
            inserted_total += int(traffic["inserted"])
            updated_total += int(expert["updated"])
            click.echo(
                "[drift-v2-online] "
                f"tick={tick} "
                f"visible_events={last_bundle.visible_event_count} "
                f"windows={last_bundle.complete_window_count} "
                f"inserted={traffic['inserted']} "
                f"expert_updated={expert['updated']} "
                f"status={last_bundle.latest_payload['labels']['status']}"
            )
            if interval_seconds > 0 and (max_ticks <= 0 or tick < max_ticks):
                time.sleep(interval_seconds)

        return {
            "run_id": config.run_id,
            "source_mode": config.source_mode,
            "ticks": tick,
            "inserted": inserted_total,
            "expert_updated": updated_total,
            "visible_events": last_bundle.visible_event_count if last_bundle else 0,
            "total_events": last_bundle.total_event_count if last_bundle else 0,
            "latest_summary": last_bundle.summary_payload if last_bundle else None,
        }


@contextmanager
def _acquire_run_lock(output_root: Path, run_id: str, mode: str):
    run_dir = output_root / f"{run_id}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".replay.lock"
    active_path = run_dir / ".active"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise click.ClickException(
                f"another drift_v2 replay is already writing to {run_dir}"
            ) from exc
        lock_file.write(f"{run_id}\n{mode}\n")
        lock_file.flush()
        active_path.write_text(f"{run_id}\n{mode}\n", encoding="utf-8")
        try:
            yield
        finally:
            if active_path.exists():
                active_path.unlink()
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
