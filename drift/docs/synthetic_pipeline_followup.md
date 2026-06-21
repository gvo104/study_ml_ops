# Follow-up по качеству synthetic pipeline

## Контекст

Эта заметка фиксирует проблемы, к которым нужно вернуться после текущего этапа с `Prometheus + Grafana`.

Базовый референсный запуск:
- `drift/artifacts/runs/20260621T192330Z_debug/`

Ключевые артефакты:
- `run_summary.json`
- `label_diagnostics.json`
- `accepted_samples.jsonl`
- `rejected_samples.jsonl`

## Что уже улучшилось

- `accepted_samples` выросли с `60` до `71`
- `rejected_samples` снизились с `130` до `71`
- `token_label_association_drift` перестал массово падать в `insufficient_data`
- `label_diagnostics.json` теперь показывает accept-rate и confusion patterns по классам

## Нерешённые проблемы

### 1. Generator -> Expert всё ещё слаб на части классов

Основные проблемные target labels:
- `Normal`
- `Bipolar`

Наблюдения по `label_diagnostics.json`:
- `Normal`:
  - `accept_rate_by_label = 0.375`
  - часто уходит в `Depression` и `Stress`
- `Bipolar`:
  - `accept_rate_by_label = 0.6`
  - часто уходит в `Anxiety` и `Stress`

Причина:
- для `Normal` генератор всё ещё иногда пишет mild-distress / low-mood формулировки
- для `Bipolar` генератор смешивает признаки с тревогой и стрессом, вместо более чистого episodic bipolar pattern

### 2. Predict model сильно расходится с expert на accepted samples

Наиболее проблемные expert labels:
- `Personality disorder`
- `Stress`
- `Suicidal`
- `Normal`

Наблюдения по `model_expert_disagreement_by_label`:
- `Personality disorder`: `1.0`
- `Normal`: `1.0`
- `Stress`: `0.8571`
- `Suicidal`: `0.7`

Это уже не проблема accept/reject-фильтра.
Это сигнал, что основная production-модель плохо переносит часть synthetic samples или слабо различает эти классы.

### 3. Baseline synthetic windows всё ещё часто дают высокий disagreement

Даже в `A_baseline` окнах виден высокий `model_expert_disagreement_rate`.

Это означает:
- либо synthetic baseline всё ещё недостаточно похож на training distribution production-модели
- либо сама production-модель имеет слабые границы между отдельными классами

## К чему вернуться позже

### Prompt-level доработки

- Для `Normal` ещё жёстче запретить формулировки:
  - `down`
  - `empty`
  - `low energy`
  - `motivation`
  - `can't enjoy things`
  - `can't shake it`
- Для `Bipolar` требовать более чистый паттерн:
  - reduced need for sleep without simple fatigue framing
  - elevated energy
  - impulsive or risky behavior
  - alternating high/low episode structure
- Ослабить смешение `Stress` и `Anxiety` в generator prompts
- Ослабить смешение `Personality disorder` и `Depression`

### Model-side разбор production classifier

- Разобрать confusion по accepted synthetic samples
- Понять, почему:
  - `Stress` часто предсказывается как `Anxiety`
  - `Personality disorder` уходит в `Stress/Anxiety`
  - `Suicidal` уходит в `Depression/Normal`
- При необходимости отдельно оценить качество production-модели на subset по этим классам

### Debug/full drift interpretation

- Не переоценивать debug-run как финальную оценку drift
- После интеграции `Prometheus + Grafana` вернуться к сравнению:
  - synthetic debug signal
  - synthetic full signal
  - устойчивость порогов на baseline окнах

## Решение на текущий момент

Эту проблему сознательно откладываем.

Текущий следующий фокус:
- интеграция `Prometheus + Grafana`
- экспорт drift metrics из `drift-runner`
- визуализация последних значений и статусов окон
