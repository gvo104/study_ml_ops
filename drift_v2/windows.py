from __future__ import annotations

from drift_v2.contracts import DemoEvent, WindowBatch, WindowPlan


def build_window_plan(
    events: list[DemoEvent],
    visible_events: int,
    window_size: int,
    step_size: int,
) -> WindowPlan:
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive")

    bounded_visible_events = max(0, min(visible_events, len(events)))
    if bounded_visible_events < window_size:
        total_complete_windows = 0
    else:
        total_complete_windows = (
            (bounded_visible_events - window_size) // step_size
        ) + 1

    return WindowPlan(
        total_events=len(events),
        visible_events=bounded_visible_events,
        window_size=window_size,
        step_size=step_size,
        total_complete_windows=total_complete_windows,
    )


def build_prefix_windows(
    events: list[DemoEvent],
    visible_events: int,
    window_size: int,
    step_size: int,
) -> list[WindowBatch]:
    plan = build_window_plan(
        events=events,
        visible_events=visible_events,
        window_size=window_size,
        step_size=step_size,
    )
    visible_prefix = events[: plan.visible_events]
    windows: list[WindowBatch] = []

    for window_index, start in enumerate(
        range(0, len(visible_prefix) - window_size + 1, step_size)
    ):
        batch_events = tuple(visible_prefix[start : start + window_size])
        windows.append(
            WindowBatch(
                window_index=window_index,
                start_event_index=batch_events[0].event_index,
                end_event_index=batch_events[-1].event_index,
                events=batch_events,
            )
        )
    return windows
