import pytest

from drift.synthetic_api.prompts import (
    build_expert_system_prompt,
    build_expert_user_prompt,
    build_generator_system_prompt,
    build_generator_user_prompt,
)
from drift.synthetic_api.schemas import ExpertRequest, GeneratorRequest


def test_generator_prompt_is_composed_from_role_and_guides():
    payload = GeneratorRequest(
        role="generator",
        phase="B",
        target_label="Anxiety",
        constraints={"length": "short", "style": "plain"},
    )

    system_prompt = build_generator_system_prompt()
    user_prompt = build_generator_user_prompt(payload)

    assert 'exactly "generator"' in system_prompt
    assert "Status guide:" in user_prompt
    assert "Phase guide:" in user_prompt
    assert "Label boundary guide:" in user_prompt
    assert "- Anxiety:" in user_prompt
    assert "- Anxiety must include:" in user_prompt
    assert "- Anxiety must avoid:" in user_prompt
    assert 'target_label="Anxiety"' in user_prompt


def test_expert_prompt_contains_only_allowed_statuses():
    payload = ExpertRequest(
        role="expert",
        text="I feel anxious and cannot sleep.",
        allowed_labels=["Anxiety", "Stress"],
    )

    system_prompt = build_expert_system_prompt()
    user_prompt = build_expert_user_prompt(payload)

    assert 'exactly "expert"' in system_prompt
    assert "- Anxiety:" in user_prompt
    assert "- Stress:" in user_prompt
    assert "- Anxiety must include:" in user_prompt
    assert "- Stress must avoid:" in user_prompt
    assert "- Bipolar:" not in user_prompt
    assert "Confidence must be a number from 0 to 1." in user_prompt
    assert 'do not choose "Normal"' in system_prompt
    assert "do not choose Normal" in user_prompt
