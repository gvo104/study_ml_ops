from drift.synthetic_api.schemas import ExpertRequest, GeneratorRequest


def build_generator_system_prompt() -> str:
    return (
        "You generate realistic short patient обращения for mental health drift "
        "simulation. Return JSON only. Do not include markdown. Do not explain."
    )


def build_generator_user_prompt(payload: GeneratorRequest) -> str:
    return (
        "Generate one patient message in JSON.\n"
        f"phase: {payload.phase}\n"
        f"target_label: {payload.target_label}\n"
        f"length: {payload.constraints.length}\n"
        f"style: {payload.constraints.style or 'neutral'}\n"
        'Return keys: role, text, target_label, phase.'
    )


def build_expert_system_prompt() -> str:
    return (
        "You are a clinical expert labeler. Choose exactly one label from the "
        "allowed list. Return JSON only with role, label, confidence, reason."
    )


def build_expert_user_prompt(payload: ExpertRequest) -> str:
    allowed_labels = ", ".join(payload.allowed_labels)
    return (
        "Classify the patient message.\n"
        f"allowed_labels: {allowed_labels}\n"
        f"text: {payload.text}"
    )


def build_repair_prompt(raw_response: str) -> str:
    return (
        "Repair the following output into valid JSON only. Preserve the intended "
        f"meaning.\nraw_output: {raw_response}"
    )
