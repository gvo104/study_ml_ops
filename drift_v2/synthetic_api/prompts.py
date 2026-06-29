from functools import lru_cache
from pathlib import Path

from drift_v2.synthetic_api.schemas import ExpertRequest, GeneratorRequest


PROMPTS_DIR = Path(__file__).resolve().parent / "prompt_templates"


@lru_cache(maxsize=None)
def _read_prompt_template(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


def _render_prompt_template(filename: str, **kwargs) -> str:
    template = _read_prompt_template(filename)
    return template.format(**kwargs).strip()


def _load_status_descriptions() -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for line in _read_prompt_template("status_definitions.txt").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        label, description = line.split(":", maxsplit=1)
        descriptions[label.strip()] = description.strip()
    return descriptions


def _load_label_boundaries() -> dict[str, dict[str, str]]:
    boundaries: dict[str, dict[str, str]] = {}
    current_label: str | None = None
    for raw_line in _read_prompt_template("label_boundaries.txt").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_label = line[1:-1].strip()
            boundaries[current_label] = {}
            continue
        if current_label is None or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        boundaries[current_label][key.strip()] = value.strip()
    return boundaries


def _build_status_guide(labels: list[str]) -> str:
    descriptions = _load_status_descriptions()
    guide_lines = []
    for label in labels:
        description = descriptions.get(label)
        if description is not None:
            guide_lines.append(f"- {label}: {description}")
    return "\n".join(guide_lines)


def _build_boundary_guide(labels: list[str]) -> str:
    boundaries = _load_label_boundaries()
    guide_lines = []
    for label in labels:
        config = boundaries.get(label)
        if config is None:
            continue
        guide_lines.append(f"- {label} must include: {config.get('must_include', '')}")
        guide_lines.append(f"- {label} must avoid: {config.get('must_avoid', '')}")
        guide_lines.append(f"- {label} contrast: {config.get('contrast', '')}")
    return "\n".join(guide_lines)


def build_generator_system_prompt() -> str:
    return _read_prompt_template("generator_system.txt")


def build_generator_user_prompt(payload: GeneratorRequest) -> str:
    return _render_prompt_template(
        "generator_user.txt",
        phase=payload.phase,
        target_label=payload.target_label,
        length=payload.constraints.length,
        style=payload.constraints.style or "neutral",
        phase_guide=_read_prompt_template("phase_definitions.txt"),
        status_guide=_build_status_guide(list(_load_status_descriptions().keys())),
        boundary_guide=_build_boundary_guide([payload.target_label]),
    )


def build_expert_system_prompt() -> str:
    return _read_prompt_template("expert_system.txt")


def build_expert_user_prompt(payload: ExpertRequest) -> str:
    return _render_prompt_template(
        "expert_user.txt",
        allowed_labels=", ".join(payload.allowed_labels),
        status_guide=_build_status_guide(payload.allowed_labels),
        boundary_guide=_build_boundary_guide(payload.allowed_labels),
        text=payload.text,
    )


def build_repair_prompt(raw_response: str) -> str:
    return _render_prompt_template(
        "repair_user.txt",
        raw_response=raw_response,
    )
