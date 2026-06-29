#!/usr/bin/env bash
set -euo pipefail

namespace="${K8S_NAMESPACE:-ml-team}"

pids=()

cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}

trap cleanup EXIT INT TERM

forward() {
  local service="$1"
  local mapping="$2"

  kubectl port-forward -n "$namespace" "svc/${service}" "$mapping" &
  pids+=("$!")
}

echo "Forwarding Kubernetes services from namespace ${namespace}:"
echo "  MinIO S3 API: http://localhost:9000"
echo "  MLflow UI:    http://localhost:5000"
echo "  Webapp:       http://localhost:8000"
echo
echo "Press Ctrl+C to stop port-forwarding."

forward minio 9000:9000
forward mlflow 5000:5000
forward webapp 8000:8000

wait
