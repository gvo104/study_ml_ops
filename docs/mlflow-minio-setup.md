
# MLflow и MinIO (S3) — инфраструктура для команды

## Обзор

В проекте настроена локальная инфраструктура для командной работы над ML-экспериментами:

| Компонент | Назначение | Адрес |
|-----------|-----------|-------|
| **MinIO** | S3-совместимое хранилище для данных и артефактов | API: `http://localhost:9000`<br>Консоль: `http://localhost:9001` |
| **MLflow** | Трекинг экспериментов, хранение метрик и моделей | UI: `http://localhost:5000` |
| **DVC** | Версионирование датасетов | Хранит данные в MinIO |

Все компоненты поднимаются одной командой через Docker Compose.

---

## Как запустить инфраструктуру

```bash
# Запуск MinIO + MLflow
make infra-up

# Проверить статус
make infra-status

# Посмотреть логи (если что-то не работает)
make infra-logs

# Остановить всё
make infra-down
```

После запуска открой в браузере:
- **MinIO Console**: http://localhost:9001 (логин: `minioadmin`, пароль: `minioadmin`)
- **MLflow UI**: http://localhost:5000

---

## Переменные окружения

Скопируй шаблон и при необходимости отредактируй:

```bash
cp .env.example .env
```

Или задай вручную:

```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_ENDPOINT_URL=http://localhost:9000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=mental_health_classification
```

Если `.env` нет, `src/config.py` подставляет те же дефолты для локального MinIO.

---

## Работа с данными через DVC

### Добавить новый датасет под контроль версий

```bash
# Добавляем файл данных
dvc add data/processed/my_dataset.csv

# Отправляем данные в S3 (MinIO)
dvc push

# Коммитим метафайл в Git
git add data/processed/my_dataset.csv.dvc
git commit -m "feat(data): add my_dataset under DVC control"
```

### Скачать данные (для второго участника)

```bash
git pull          # получить .dvc-метафайлы
dvc pull          # скачать сами данные из MinIO
```

### Проверить статус данных

```bash
dvc status        # покажет, есть ли изменения в данных
```

---

## Работа с MLflow

Логирование встроено в `src/models/trainer.py` через `src/models/tracking.py`.
Вручную писать `mlflow.start_run()` не нужно.

### Production-обучение (локальные артефакты + MLflow)

```bash
make infra-up
make train
```

Конфиг: `configs/xgboost_baseline.yaml`. Эксперимент MLflow:
`mental_health_classification`. Run name: `xgboost_baseline`.

### Пакет экспериментов (только MLflow, без перезаписи `models/xgboost/`)

```bash
make experiments
```

Читает все `configs/experiments/*.yaml` (XGBoost, LightGBM, Logistic Regression,
Random Forest). В конце выводит таблицу accuracy / macro_f1 и лучший run.

Один конфиг:

```bash
python -m src.models.train_model --config configs/experiments/xgboost_deep.yaml
```

### Что попадает в каждый MLflow run

| Путь в Artifacts | Содержимое |
|------------------|------------|
| `model/` | Обученная модель (flavor: xgboost / lightgbm / sklearn) |
| `evaluation/confusion_matrix.png` | Матрица ошибок |
| `evaluation/classification_report.txt` | Отчёт sklearn |
| `model_bundle/` | Локальные pkl (только если `save_local_artifacts: true`) |

Метрики: `accuracy`, `macro_f1`, `weighted_f1`. Параметры — из YAML-конфига.

### Просмотр и сравнение

1. Открой http://localhost:5000
2. Experiment: `mental_health_classification`
3. Сортируй по `macro_f1` или `accuracy`
4. Artifacts → `evaluation/` → confusion matrix

### Новый эксперимент

Скопируй `configs/experiments/xgboost_baseline.yaml`, задай уникальный
`experiment.run_name` и параметры `model` / `features`.

### Регистрация модели (опционально)

```bash
# Запусти MLflow с поддержкой реестра моделей (добавь флаг при старте)
# --registry-store-uri sqlite:///path/to/registry.db
```

---

## Структура бакета в MinIO

```
ml-team/
├── dvc-store/          # Данные под управлением DVC (хешированные)
│   └── files/md5/...
└── mlflow-artifacts/   # Артефакты MLflow (модели, метрики, графики)
    └── <experiment_id>/
        └── <run_id>/
            └── artifacts/
```

---

## Типовой рабочий процесс для команды

### Первый участник (тот, кто поднимает инфраструктуру)

1. `make infra-up` — запустить MinIO и MLflow
2. Обработать данные и добавить под DVC:
   ```bash
   dvc add data/processed/fin_data.csv
   dvc push
   git add data/processed/fin_data.csv.dvc
   git commit -m "feat(data): add fin_data"
   git push
   ```
3. Запустить обучение и/или sweep экспериментов:
   ```bash
   make train
   make experiments
   ```
4. Результаты смотреть в MLflow UI (вкладка Artifacts → `evaluation/`)

### Второй участник (подключается удалённо)

1. `git pull` — получить актуальный код и DVC-метафайлы
2. `dvc pull` — скачать данные из MinIO (нужен доступ по IP)
3. `make train` — запустить обучение
4. Результаты автоматически попадают в общий MLflow и MinIO

---

## Удалённый доступ для второго участника

Если MinIO и MLflow запущены на машине первого участника (IP: `192.168.x.x`):

1. Открыть порты в брандмауэре Windows:
   ```powershell
   New-NetFirewallRule -DisplayName "MinIO API" -Direction Inbound -LocalPort 9000 -Protocol TCP -Action Allow
   New-NetFirewallRule -DisplayName "MinIO Console" -Direction Inbound -LocalPort 9001 -Protocol TCP -Action Allow
   New-NetFirewallRule -DisplayName "MLflow UI" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow
   ```

2. Второй участник устанавливает переменные окружения с IP первого:
   ```bash
   export AWS_ENDPOINT_URL=http://192.168.x.x:9000
   export MLFLOW_S3_ENDPOINT_URL=http://192.168.x.x:9000
   export MLFLOW_TRACKING_URI=http://192.168.x.x:5000
   ```

3. Открывает в браузере:
   - MinIO: `http://192.168.x.x:9001`
   - MLflow: `http://192.168.x.x:5000`

---

## Устранение неполадок

### Контейнер не стартует (ошибка "name already in use")

```bash
docker stop minio mlflow minio-init
docker rm minio mlflow minio-init
make infra-up
```

### DVC не видит S3 (ImportError: dvc-s3)

```bash
pip install dvc[s3]
```

### Файл `.dvc` игнорируется Git

Проверить правило:
```bash
git check-ignore -v data/processed/my_file.csv.dvc
```
Добавить в `.gitignore`: `!data/**/*.dvc`

### AWS CLI не видит креды

Передать явно:
```bash
AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin aws --endpoint-url http://localhost:9000 s3 ls s3://ml-team/
```

### `make train` падает на MLflow / NoCredentialsError

1. Убедись, что `make infra-up` запущен.
2. Скопируй `.env.example` → `.env` или полагайся на дефолты в `src/config.py`.
3. Обучение всё равно сохранит локальные артефакты в `models/xgboost/`;
   при ошибке S3 в логе будет warning, а не crash (после последнего обновления `tracking.py`).

---

## Полезные команды

```bash
# Посмотреть содержимое бакета
aws --endpoint-url http://localhost:9000 s3 ls s3://ml-team/ --recursive

# Очистить MLflow-артефакты
aws --endpoint-url http://localhost:9000 s3 rm s3://ml-team/mlflow-artifacts/ --recursive

# Статус DVC
dvc status

# Список отслеживаемых файлов
dvc list . --dvc-only
```

---

## Файлы, относящиеся к инфраструктуре

| Файл | Назначение |
|------|-----------|
| `docker-compose.yml` | Описание сервисов MinIO и MLflow |
| `Dockerfile.mlflow` | Сборка образа для MLflow |
| `Makefile` | Команды `infra-up/down/status/logs` |
| `.dvc/config` | Настройка S3 remote для DVC |
| `.gitignore` | Правила игнорирования данных и артефактов |