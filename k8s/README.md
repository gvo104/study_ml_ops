# K8s Infrastructure (Equivalent to docker-compose)

This directory contains Kubernetes manifests for deploying MinIO, MLflow and the API web application, similar to the existing `docker-compose.yml`.

## Architecture

```
┌─────────────────┐       ┌──────────────────┐
│     MinIO       │──────▶│  Bucket: ml-team │
│   (S3 Storage)  │       │                  │
└─────────────────┘       └──────────────────┘
         ▲                           |
         |                           v
         |                 ┌──────────────────┐
         └────────────────▶│     MLflow       │
                           │ (Tracking Server)│
                           └──────────────────┘
                                    ▲
                                    |
                           ┌──────────────────┐
                           │      Webapp      │
                           │   (FastAPI UI)   │
                           └──────────────────┘
```

## Directory Structure

```
k8s/
├── base/                    # Base Kubernetes manifests
│   ├── configmap.yml       # Configuration values
│   ├── minio.yml           # MinIO deployment & service
│   ├── mlflow.yml          # MLflow deployment, init job & service
│   └── webapp.yml          # API deployment & service
└── overlays/               # Environment-specific configs (optional)
    ├── dev/                # Dev environment overrides
    └── production/         # Production environment overrides
```

## Prerequisites

- Kubernetes cluster (minikube, k3d, EKS, GKE, etc.)
- `kubectl` installed and configured
- Docker images available in the cluster:
  - `mlflow:latest` built from `Dockerfile.mlflow`
  - `study-mlops-webapp:latest` built from `Dockerfile.api`
- Argo CD installed in your cluster (optional)
- kustomize (optional, for applying manifests)

## Quick Start

### Using kubectl directly

```bash
# Build local images used by the Kubernetes manifests
docker build -f Dockerfile.mlflow -t mlflow:latest .
docker build -f Dockerfile.api -t study-mlops-webapp:latest .

# For kind, load images into the cluster
kind load docker-image mlflow:latest --name ml-cluster
kind load docker-image study-mlops-webapp:latest --name ml-cluster

# For minikube, either build inside minikube's Docker daemon or load images:
minikube image load mlflow:latest
minikube image load study-mlops-webapp:latest

# Apply all manifests
kubectl apply -k k8s/base/

# Check status
kubectl get pods -n ml-team

# Access services
kubectl port-forward -n ml-team svc/webapp 8000:8000
kubectl port-forward -n ml-team svc/mlflow 5000:5000
kubectl port-forward -n ml-team svc/minio 9001:9001
```

### Using Kustomize

```bash
# Build and apply
kubectl kustomize k8s/base | kubectl apply -f -

# Or with local kustomize
kustomize build k8s/base | kubectl apply -f -
```

### Using Argo CD (Recommended)

1. Deploy the `ml-team-app` to your cluster via Argo CD UI:
   - Open Argo CD dashboard
   - Click "Add Application"
   - Fill in the details from [../arcd/ml-team-application.yml](../arcd/ml-team-application.yml)

2. Push changes to deploy automatically:
   ```bash
   # Modify k8s/base/ files
   git commit -m "Update infrastructure"
   git push origin main
   ```

Argo CD will automatically sync and update the cluster.

## Services

### MinIO (S3 Storage)

| Port  | Service    | Description          |
|-------|------------|----------------------|
| 9000  | S3 API     | Object storage API   |
| 9001  | Console    | Web UI for MinIO     |

**Credentials:**
- Username: `minioadmin`
- Password: `minioadmin`

### MLflow Tracking Server

| Port | Service | Description      |
|------|---------|------------------|
| 5000 | Server  | MLflow tracking  |

**Environment:**
- Backend: SQLite (local) or S3 (via MinIO)
- Artifact storage: `s3://ml-team/mlflow-artifacts` (MinIO)

### Webapp API

| Port | Service | Description |
|------|---------|-------------|
| 8000 | HTTP    | FastAPI UI and prediction API |

**Environment:**
- `MLFLOW_TRACKING_URI`: `http://mlflow:5000`
- `MLFLOW_ENABLED`: `true`
- S3/DVC/MLflow artifact endpoint: `http://minio:9000`

## Configuring Kubernetes Locally

### Using k3d (Recommended for development)

```bash
# Start a local cluster
k3d cluster create ml-cluster --agents 1

# Apply manifests
kubectl apply -k k8s/base/

# Get access to cluster
k3d kubeconfig export ml-cluster
```

### Using minikube

```bash
# Start minikube
minikube start --memory 4096 --cpus 2

# Load into Docker for images
minikube docker-env

# Apply manifests
kubectl apply -k k8s/base/

# Exit Docker env
exit
```

### Using kind (Kubernetes in Docker)

```bash
# Create cluster
kind create cluster --name ml-cluster

# Apply with Kubernetes YAML
kubectl apply -k k8s/base/

# Or using kustomize
kustomize build k8s/base | kubectl apply -f -
```

## Updating Infrastructure

When you change `k8s/base/` files:

1. **Manual update:**
   ```bash
   # Check Argo CD status
   argocd app get ml-team-app
   
   # Force sync (if needed)
   argocd app set --sync-only ml-team-app
   ```

2. **Automatic update (via CI/CD):**
   - Push changes to `main` branch
   - GitHub Actions workflow will deploy to Argo CD
   - Argo CD will automatically sync to Kubernetes cluster

## Monitoring

```bash
# View all resources in ml-team namespace
kubectl get all -n ml-team

# View pods
kubectl get pods -n ml-team

# View logs for specific pod
kubectl logs -n ml-team <pod-name>

# Describe deployment
kubectl describe deployment mlflow -n ml-team

# Watch events
kubectl get events -n ml-team --watch
```

## Troubleshooting

### MinIO not starting

```bash
# Check pod status
kubectl get pods -n ml-team

# View logs
kubectl logs -n ml-team minio-xxxxx-x
```

Common issues:
- Insufficient memory/resources
- Missing secret configuration
- Network policies blocking access

### MLflow not connecting to MinIO

Ensure ConfigMap `minio-config` has correct values:
```bash
kubectl get configmap minio-config -n ml-team -o yaml
```

## Environment Variables Reference

| Variable | Used By | Default Value |
|----------|---------|---------------|
| `MINIO_ROOT_USER` | MinIO, MLflow | `minioadmin` |
| `MINIO_ROOT_PASSWORD` | MinIO, MLflow | `minioadmin` |
| `AWS_ACCESS_KEY_ID` | MLflow | (from configmap) |
| `AWS_SECRET_ACCESS_KEY` | MLflow | (from configmap) |
| `MLFLOW_S3_ENDPOINT_URL` | MLflow | `http://minio:9000` |
| `AWS_DEFAULT_REGION` | MLflow | `us-east-1` |

## Migration from Docker Compose

The Kubernetes setup is designed to be equivalent to your existing `docker-compose.yml`:

| docker-compose           | Kubernetes          | Notes                        |
|--------------------------|---------------------|------------------------------|
| `version: '3.8'`         | N/A                 | K8s uses API versions        |
| `services.minio.image`   | `deployment.image`  | Container image              |
| `ports`                  | `service.ports`     | ClusterIP services           |
| `environment`            | `env` / `configmap` | ConfigMap for configs        |
| `depends_on`             | Init Job           | minio-init Job before MLflow |
| `volumes`                | `PersistentVolumeClaim` | Persistent data for MinIO, MLflow, API models/data |

## Next Steps

- Add Ingress Controller for external access
- Set up monitoring with Prometheus/Grafana
- Add certificate management for HTTPS
- Implement RBAC for role-based access control
