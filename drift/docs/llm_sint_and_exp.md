# Synthetic API: generator and expert

## Current Role

`synthetic-api` - отдельный FastAPI сервис для LLM-based synthetic traffic.

Он обслуживает один endpoint:

```text
POST /llm/run
```

Роли:

- `generator`: генерирует текст пользовательского обращения по phase и hidden target label;
- `expert`: размечает готовый текст, не получая phase, target label или synthetic metadata.

Основной `predict-api` не загружает LLM и не зависит от `synthetic-api`.

## Runtime

В production-like локальном режиме ожидается одна локальная модель `Qwen2.5-3B-Instruct GGUF` через `llama.cpp`.

Вызовы сериализуются через lock, чтобы на одной машине не было параллельного LLM inference.

Для tests/dev есть `MockLlmRuntime`, который повторяет контракт без загрузки модели.

## Generator Contract

Request:

```json
{
  "role": "generator",
  "phase": "A|B|C|D",
  "target_label": "Anxiety|Bipolar|Depression|Normal|Personality disorder|Stress|Suicidal",
  "constraints": {
    "style": "neutral",
    "length": "short|medium|long"
  }
}
```

Response:

```json
{
  "role": "generator",
  "text": "generated patient message",
  "target_label": "Stress",
  "phase": "B"
}
```

`phase` и `target_label` являются orchestration metadata. Они не передаются в expert request и не экспортируются как monitoring signal.

## Expert Contract

Request:

```json
{
  "role": "expert",
  "text": "patient message",
  "allowed_labels": [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Normal",
    "Personality disorder",
    "Stress",
    "Suicidal"
  ]
}
```

Response:

```json
{
  "role": "expert",
  "label": "Stress",
  "confidence": 0.82,
  "reason": "short concrete reason"
}
```

Expert считается proxy truth для demo monitoring. Reject/regenerate больше не является runtime-механизмом нового pipeline.

Если expert системно ошибается, чинить нужно prompt/boundary guide или replay cache, а не скрыто отбрасывать события в monitoring flow.

## Phase Prompting

Phase guide живет в:

```text
drift/synthetic_api/prompt_templates/phase_definitions.txt
```

Текущий смысл фаз:

- `A`: baseline-like wording;
- `B`: lexical/style shift при той же семантике;
- `C`: baseline-like wording, distribution shift создается orchestrator;
- `D`: context-based token-label association shift.

Для `D` expert prompt явно требует интерпретировать marker words через контекст, а не как single keyword.

## Current Orchestration

Основной orchestrator:

```text
drift.demo.phase_worker
```

Flow:

```text
generator -> predict-api -> expert -> SQLite
```

После генерации текст обязательно проходит production preprocessing через `drift.runner.pipeline.build_preprocessed_record`.

## Required Checks

- `generator` возвращает strict JSON с `role`, `text`, `target_label`, `phase`;
- `expert` возвращает strict JSON с `role`, `label`, `confidence`, `reason`;
- expert request не принимает `phase` и `target_label`;
- `predict-api` не загружает LLM;
- parallel requests к real LLM runtime сериализуются;
- phase-specific prompt instructions покрыты tests.
