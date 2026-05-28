# Контекст проекта study_ml_ops

Документ фиксирует текущее состояние репозитория перед переносом логики из отдельной структуры `pipeline/` в шаблон проекта `study_ml_ops/`.

## Цель ближайшего этапа

В папке `study_ml_ops/` лежит шаблон проекта в стиле cookiecutter data science. В него планируется перенести уже реализованную ML-логику, которая сейчас находится отдельно:

- `pipeline/` - текущий рабочий пайплайн обучения и инференса.
- корень проекта - данные, окружение, ноутбуки, MLflow DB и сохраненные модели.
- `study_ml_ops/` - целевая структура проекта, пока в основном шаблонная.

На текущем шаге перенос логики не выполняется. Этот файл нужен как карта проекта и опорный контекст для следующей итерации.

## Текущая структура

```text
.
├── pipeline/                 # рабочая ML-логика
├── study_ml_ops/             # шаблон целевого проекта
├── data/                     # локальные CSV и ноутбук с подготовкой данных
├── saved_models/xgboost/     # сохраненные артефакты модели
├── logs/                     # локальные логи
├── environment.yml           # conda-окружение
├── mlflow.db                 # локальная база MLflow
├── 1.ipynb                   # крупный ноутбук с экспериментами
└── .gitignore                # корневые исключения для артефактов
```

## Целевая структура `study_ml_ops/`

`study_ml_ops/` содержит стандартный шаблон:

- `src/data/make_dataset.py` - заготовка для подготовки данных.
- `src/features/build_features.py` - пустой модуль для feature engineering.
- `src/models/train_model.py` - пустой модуль для обучения.
- `src/models/predict_model.py` - пустой модуль для предсказаний.
- `src/visualization/visualize.py` - заготовка под визуализации.
- `data/raw`, `data/interim`, `data/processed`, `data/external` - целевые папки данных.
- `models/` - целевая папка для сериализованных моделей.
- `docs/`, `reports/`, `references/`, `notebooks/` - стандартные папки проекта.

`study_ml_ops/README.md` уже описывает проект как `Production-ready MLOps system for mental health text classification`.

Важно: `study_ml_ops/setup.py` сейчас объявляет пакет с именем `src`, что соответствует шаблону, но при переносе можно будет решить, оставлять ли такой импортный путь или переименовать пакет более явно.

## Текущая ML-логика в `pipeline/`

### Конфигурация

Файл: `pipeline/config.py`

- `BASE_DIR` - корень репозитория.
- `DATA_PATH` - `data/fin_data.csv`.
- `MODEL_DIR` - `saved_models/xgboost`.
- `RANDOM_STATE` - `101`.
- `TEST_SIZE` - `0.2`.
- `TFIDF_MAX_FEATURES` - `5000`.
- `NGRAM_RANGE` - `(1, 2)`.
- `SVD_COMPONENTS` - `300`.

Часть значений из конфига дублируется прямо в коде `FeatureBuilder` и `train.py`; при переносе стоит централизовать их.

### Загрузка данных

Файл: `pipeline/load_data.py`

Логика:

- читает CSV из `DATA_PATH`;
- использует `index_col=0`;
- удаляет строки с пропусками через `dropna()`;
- сбрасывает индекс.

Ожидаемый датасет: `data/fin_data.csv`.

### Предобработка текста

Файл: `pipeline/preprocess.py`

Логика:

- приводит текст к нижнему регистру;
- удаляет URL;
- удаляет mentions вида `@user`;
- удаляет пунктуацию через regex `[^\w\s]`;
- токенизирует через `nltk.word_tokenize`;
- применяет `PorterStemmer`;
- поддерживает параллельную обработку через `multiprocessing.Pool`.

Особенность: при импорте выполняется `nltk.download("punkt", quiet=True)`. При переносе лучше вынести загрузку NLTK-ресурсов из импорт-сайда эффекта.

### Feature engineering

Файл: `pipeline/feature_engineering.py`

Класс: `FeatureBuilder`

Логика:

- строит `TfidfVectorizer` с `ngram_range=(1, 2)` и `max_features=5000`;
- снижает размерность через `TruncatedSVD(n_components=300, random_state=101)`;
- объединяет SVD-признаки с числовыми признаками через `np.hstack`;
- поддерживает `fit_transform` и `transform`.

Числовые признаки в обучении:

- `num_of_characters`;
- `num_of_sentences`.

### Обучение

Файл: `pipeline/train.py`

Основной поток:

1. загрузка датасета;
2. предобработка `df["tokens_stemmed"]` в `processed_text`;
3. выделение текстовых и числовых признаков;
4. кодирование таргета `status` через `LabelEncoder`;
5. `train_test_split(test_size=0.2, random_state=101)`;
6. построение TF-IDF + SVD + числовых признаков;
7. балансировка train-выборки через `RandomOverSampler(random_state=101)`;
8. обучение `XGBClassifier`;
9. оценка через `accuracy_score` и `classification_report`;
10. сохранение модели, feature artifacts, encoder и metadata.

Модель:

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=101,
    eval_metric="mlogloss",
)
```

Ожидаемые колонки датасета:

- `tokens_stemmed`;
- `num_of_characters`;
- `num_of_sentences`;
- `status`.

### Сохранение артефактов

Файл: `pipeline/save_artifacts.py`

Сохраняет в `saved_models/xgboost/`:

- `xgboost_model.pkl`;
- `vectorizer.pkl`;
- `svd.pkl`;
- `label_encoder.pkl`;
- `metadata.json`.

В `metadata.json` записываются:

- `accuracy`;
- `created_at`;
- `classes`.

Текущая metadata:

- accuracy: `0.7759111617312073`;
- created_at: `2026-05-27 21:03:31.206669`;
- classes: `Anxiety`, `Bipolar`, `Depression`, `Normal`, `Personality disorder`, `Stress`, `Suicidal`.

В папке также есть `pca.pkl` и `versions.json`; они не используются текущим кодом `pipeline/inference.py`, но могут быть артефактами предыдущих экспериментов.

### Инференс

Файл: `pipeline/inference.py`

Класс: `MentalHealthPredictor`

Логика:

- загружает модель, vectorizer, SVD, label encoder и metadata;
- предобрабатывает входной текст через `preprocess_single`;
- строит TF-IDF + SVD;
- вычисляет числовые признаки на лету:
  - `num_chars = len(text)`;
  - `num_sentences = len(re.findall(r"[.!?]+", text)) or 1`;
- объединяет признаки;
- возвращает:
  - `prediction`;
  - `confidence`;
  - `probabilities`.

Важно для будущего переноса: train использует готовые числовые колонки из датасета, inference вычисляет их сам. Нужно проверить, совпадает ли способ расчета с подготовкой исходного датасета.

### Логирование

Файл: `pipeline/utils.py`

Содержит простой `get_logger`, который вызывает `logging.basicConfig` с форматом:

```text
%(asctime)s | %(levelname)s | %(message)s
```

## Данные и локальные артефакты

Локально обнаружены:

- `data/fin_data.csv` - основной CSV для обучения, примерно 136 MB.
- `data/prepare_data.csv` - CSV такого же размера, вероятно промежуточный или дубликат подготовленного датасета.
- `data/check_list.ipynb` - небольшой ноутбук.
- `1.ipynb` - крупный ноутбук с экспериментами.
- `mlflow.db` - локальная база MLflow, примерно 1.2 MB.
- `saved_models/xgboost/` - текущие сохраненные артефакты модели.

Корневой `.gitignore` исключает:

- `.venv/`;
- `*.csv`;
- `*.pyc`;
- `*.pkl`;
- `*.json`;
- `*.db`.

Это значит, что данные, модели, JSON metadata и MLflow DB сейчас не должны попадать в git. При переносе в `study_ml_ops/` надо сохранить это правило или аккуратно объединить его с шаблонным `.gitignore`.

## Зависимости

Фактически используемые текущим пайплайном:

- `pandas`;
- `numpy`;
- `nltk`;
- `scikit-learn`;
- `imbalanced-learn`;
- `xgboost`;
- `joblib`.

В `environment.yml` есть большое conda-окружение, включая DVC, MLflow-смежные зависимости, FastAPI/Flask, Jupyter и другие инструменты.

В `study_ml_ops/requirements.txt` пока только шаблонные зависимости:

- `click`;
- `Sphinx`;
- `coverage`;
- `awscli`;
- `flake8`;
- `python-dotenv>=0.5.1`.

При переносе нужно синхронизировать зависимости шаблона с реально используемым пайплайном.

## Предварительная карта переноса

Возможное соответствие текущих файлов целевой структуре:

- `pipeline/load_data.py` -> `study_ml_ops/src/data/make_dataset.py` или отдельный модуль в `src/data/`.
- `pipeline/preprocess.py` -> `study_ml_ops/src/features/` или `src/data/`, в зависимости от того, считать ли очистку текста подготовкой данных или построением признаков.
- `pipeline/feature_engineering.py` -> `study_ml_ops/src/features/build_features.py`.
- `pipeline/train.py` -> `study_ml_ops/src/models/train_model.py`.
- `pipeline/inference.py` -> `study_ml_ops/src/models/predict_model.py`.
- `pipeline/save_artifacts.py` -> `study_ml_ops/src/models/` или отдельный utility-модуль для model registry/artifacts.
- `pipeline/config.py` -> отдельный config-модуль внутри `study_ml_ops/src/` или конфиг в `study_ml_ops/`.
- `pipeline/utils.py` -> общий utility-модуль внутри `study_ml_ops/src/`.

## Вопросы для следующей итерации

- Нужен ли пакет с импортным именем `src`, как в cookiecutter-шаблоне, или лучше сделать явное имя пакета, например `study_ml_ops`?
- Где должна лежать рабочая копия данных после переноса: в корневом `data/` или в `study_ml_ops/data/`?
- Нужно ли переносить `saved_models/xgboost/` в `study_ml_ops/models/` или оставить модели как локальные артефакты вне шаблона?
- Планируется ли использовать MLflow/DVC в ближайшей версии, или сначала переносим только базовый train/predict pipeline?
- Нужно ли превращать обучение и инференс в CLI-команды через `click`, раз шаблон уже использует `click` в `make_dataset.py`?

## Риски и заметки

- `nltk.download("punkt")` на импорте может ломать воспроизводимость в средах без сети.
- Параллельный `multiprocessing.Pool` в предобработке может быть неудобен для Windows/Jupyter и требует аккуратного CLI-входа.
- Конфигурационные значения частично дублируются в коде.
- `pipeline/inference.py` не использует `pca.pkl`, хотя такой артефакт есть в папке модели.
- Корневой `.gitignore` исключает все `*.json`, что может случайно скрывать полезные конфиги, если они появятся.
- `study_ml_ops/requirements.txt` не отражает реальные ML-зависимости.
- `study_ml_ops/src/models/train_model.py`, `predict_model.py` и `src/features/build_features.py` сейчас пустые, поэтому перенос можно делать без риска сломать существующую логику шаблона.
