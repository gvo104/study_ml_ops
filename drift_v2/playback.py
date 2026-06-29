from __future__ import annotations

from drift_v2.contracts import PlaybackState


def initial_playback_state(total_events: int, batch_size: int) -> PlaybackState:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return PlaybackState(
        visible_events=0,
        total_events=total_events,
        batch_size=batch_size,
        cycles_completed=0,
    )


def advance_playback_state(state: PlaybackState) -> PlaybackState:
    if state.total_events <= 0:
        return state

    if state.visible_events >= state.total_events:
        next_visible = min(state.batch_size, state.total_events)
        return PlaybackState(
            visible_events=next_visible,
            total_events=state.total_events,
            batch_size=state.batch_size,
            cycles_completed=state.cycles_completed + 1,
        )

    next_visible = min(state.visible_events + state.batch_size, state.total_events)
    return PlaybackState(
        visible_events=next_visible,
        total_events=state.total_events,
        batch_size=state.batch_size,
        cycles_completed=state.cycles_completed,
    )
