
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

Перед работой установи переменные (или добавь в `~/.bashrc`):

```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_ENDPOINT_URL=http://localhost:9000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export MLFLOW_TRACKING_URI=http://localhost:5000
```

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

### Настройка в Python-коде

В начало скрипта обучения (`train_model.py`) добавь:

```python
import mlflow
import mlflow.xgboost  # или другой фреймворк

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("название_эксперимента")
```

### Логирование эксперимента

```python
with mlflow.start_run():
    # Параметры
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.01)

    # Обучение модели
    model = train_model(...)

    # Метрики
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("f1_score", 0.83)

    # Модель
    mlflow.xgboost.log_model(model, "model")

    # Дополнительные артефакты
    mlflow.log_artifact("vectorizer.pkl")
```

### Просмотр экспериментов

1. Открой MLflow UI: http://localhost:5000
2. Выбери эксперимент в левом меню
3. Сравнивай запуски по метрикам
4. Скачивай артефакты (модели, графики) из вкладки Artifacts

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
3. Запустить обучение:
   ```bash
   make train
   ```
4. Результаты смотреть в MLflow UI

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
```

---

## Как добавить в проект

Создай файл:

```bash
touch docs/mlflow-minio-setup.md
```

Скопируй содержимое выше в этот файл. Затем закоммить:

```bash
git add docs/mlflow-minio-setup.md
git commit -m "docs: add MLflow and MinIO setup guide"
```