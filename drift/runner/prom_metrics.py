from __future__ import annotations

from typing import Any


STATUS_TO_CODE = {
    "ok": 0,
    "warning": 1,
    "critical": 2,
}


def build_prom_metrics_latest(
    run_id: str,
    mode: str,
    latest_window: dict[str, Any] | None,
) -> dict[str, Any]:
    if latest_window is None:
        return {
            "run_id": run_id,
            "drift_runner_mode": mode,
            "labels": {"run_id": run_id, "mode": mode},
        }

    status = latest_window.get("overall_status", "ok")
    return {
        "run_id": run_id,
        "drift_runner_mode": mode,
        "drift_token_distribution_jsd": latest_window.get("token_distribution_jsd"),
        "drift_target_distribution_jsd": latest_window.get("target_distribution_jsd"),
        "drift_model_expert_disagreement_rate": latest_window.get(
            "model_expert_disagreement_rate"
        ),
        "drift_model_expert_macro_f1": latest_window.get("model_expert_macro_f1"),
        "drift_token_label_association_drift": latest_window.get(
            "token_label_association_drift"
        ),
        "drift_window_status_code": STATUS_TO_CODE.get(status, 0),
        "drift_window_phase": latest_window.get("phase"),
        "drift_window_index": latest_window.get("window_index"),
        "drift_insufficient_data_flag": int(
            latest_window.get("token_label_association_status") == "insufficient_data"
        ),
        "labels": {
            "run_id": run_id,
            "mode": mode,
            "phase": latest_window.get("phase"),
            "status": status,
        },
    }
