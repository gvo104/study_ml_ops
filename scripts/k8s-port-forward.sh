#!/usr/bin/env bash
set -euo pipefail

namespace="${K8S_NAMESPACE:-ml-team}"
address="${KUBECTL_PORT_FORWARD_ADDRESS:-127.0.0.1}"

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
  local namespace="$1"
  local service="$2"
  local mapping="$3"

  kubectl port-forward --address "$address" -n "$namespace" "svc/${service}" "$mapping" &
  pids+=("$!")
}

forward_if_available() {
  local namespace="$1"
  local service="$2"
  local mapping="$3"
  local label="$4"
  local url="$5"

  if kubectl get svc -n "$namespace" "$service" >/dev/null 2>&1; then
    printf "  %-13s %s\n" "${label}:" "$url"
    forward "$namespace" "$service" "$mapping"
  else
    printf "  %-13s %s\n" "${label}:" "not found (skipping svc/${service} in namespace ${namespace})"
  fi
}

echo "Forwarding Kubernetes services:"
forward_if_available "$namespace" minio 9000:9000 "MinIO S3 API" "http://${address}:9000"
forward_if_available "$namespace" mlflow 5000:5000 "MLflow UI" "http://${address}:5000"
forward_if_available "$namespace" webapp 8000:8000 "Webapp" "http://${address}:8000"
echo
echo "Press Ctrl+C to stop port-forwarding."

wait
