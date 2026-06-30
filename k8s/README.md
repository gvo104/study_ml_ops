# Kubernetes infrastructure

Каталог `k8s/` содержит Kubernetes manifests для инфраструктурной части проекта `study_ml_ops`. На текущий момент это не полный деплой всего приложения, а набор ресурсов для MinIO и MLflow с заготовкой GitOps-интеграции через Argo CD.

## Что здесь есть

В `k8s/base/` находятся:

- `configmap.yml`
  namespace, ConfigMap и Secret с базовыми значениями для MinIO/MLflow;
- `minio.yml`
  Deployment и Service для MinIO;
- `mlflow.yml`
  Job `minio-init` и Deployment/Service для MLflow;
- `kustomization.yml`
  базовый Kustomize entrypoint.

В `../arcd/ml-team-application.yml` находится Argo CD Application, указывающий на `k8s/base`.

## Что здесь отсутствует

Сейчас в каталоге нет Kubernetes manifests для:

- основного FastAPI inference/UI сервиса;
- drift monitoring stack;
- Prometheus и Grafana;
- ingress, TLS и production networking;
- persistent volumes для production storage;
- полноценной secrets strategy.

Поэтому этот каталог нужно воспринимать как частичную инфраструктурную основу, а не как завершенный Kubernetes deployment всего проекта.

## Архитектура

Текущий Kubernetes-контур покрывает только связку:

```text
MinIO (S3-compatible storage)
        ↓
   bucket ml-team
        ↓
     MLflow
```

Именно эта часть повторяет локальную связку из `docker-compose.yml`.

## Быстрый запуск

### Через kubectl

```bash
kubectl apply -f k8s/base/
kubectl get all -n ml-team
```

### Через Kustomize

```bash
kubectl kustomize k8s/base | kubectl apply -f -
```

Или:

```bash
kustomize build k8s/base | kubectl apply -f -
```

## Сервисы

### MinIO

- Service: `minio`
- Ports:
  - `9000` — S3 API
  - `9001` — console

### MLflow

- Service: `mlflow`
- Port:
  - `5000` — tracking server

MLflow использует:

- S3 endpoint: `http://minio:9000`
- bucket: `ml-team`
- backend volume: `emptyDir`

## Argo CD

Файл `arcd/ml-team-application.yml` описывает Argo CD Application для `k8s/base`.

Важно:

- это GitOps-заготовка инфраструктуры;
- она не покрывает деплой FastAPI/UI сервиса;
- для реальной эксплуатации потребуется валидная Argo CD инсталляция, корректные repo settings и рабочие secrets.

## CI/CD и Kubernetes

В `.github/workflows/deploy.yml` есть попытка интеграции с Argo CD. Однако на текущем состоянии репозитория этот workflow нельзя считать полностью готовым production CD:

- деплой относится к инфраструктурным манифестам, а не ко всему приложению;
- секреты и переменные окружения требуют реальной настройки;
- сам workflow выглядит как заготовка и требует дополнительной валидации.

## Мониторинг и troubleshooting

Проверить ресурсы:

```bash
kubectl get all -n ml-team
```

Посмотреть логи:

```bash
kubectl logs -n ml-team deployment/minio
kubectl logs -n ml-team deployment/mlflow
```

Проверить Job инициализации bucket:

```bash
kubectl get jobs -n ml-team
kubectl logs -n ml-team job/minio-init
```

## Ограничения

- используется `emptyDir`, а не постоянное хранилище;
- нет manifests для основного API/UI контейнера;
- нет отдельного deployment для monitoring stack;
- нет production ingress и сетевой конфигурации;
- нет полноценной Kubernetes-операционки вокруг модели и переобучения.

## Когда использовать этот каталог

Этот каталог полезен, если нужно:

- показать, как MinIO и MLflow могут быть развернуты в Kubernetes;
- подготовить базу для дальнейшего GitOps-деплоя;
- развивать инфраструктуру дальше до полноценного Minikube/Kubernetes решения.

Если цель — запуск проекта целиком локально, основной поддерживаемый сценарий остается за `docker-compose.yml` и `docker-compose.monitoring.yml`.
