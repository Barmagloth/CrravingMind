# CravingMind — «Жаждущий Разум»

**Версия спецификации:** 1.7.0 | **Статус:** Рабочий прототип (первые экспериментальные прогоны)
**Парадигма:** Вычислительный дарвинизм. Оптимизация через аппаратный дефицит.

---

## Что это

CravingMind — исследовательская система, в которой LLM-агент работает в условиях жёсткого дефицита ресурсов и вынужден оптимизировать собственное поведение, чтобы выжить. Агент не получает инструкций по самосовершенствованию — он получает ограниченный бюджет токенов, набор задач и четыре числа обратной связи. Всё остальное — его проблема.

**Цель проекта — не сам агент (он расходный материал), а артефакт**, который агент создаёт в процессе выживания: Python-функция `compress(text, target_ratio) → compressed_text`, способная сжимать произвольный текст с минимальной потерей смысла без единого LLM-вызова. Функция встраивается в любой пайплайн с нулевой стоимостью инференса.

---

## Быстрый старт

```bash
# Установка
pip install -e .

# Генерация бенчмарка (один раз)
python -m craving_mind.benchmark.generator --config config/default.yaml

# Запуск эксперимента (CLI провайдер — без API-ключа)
python -m craving_mind --config config/default.yaml --provider cli

# Запуск с Anthropic API
ANTHROPIC_API_KEY=sk-... python -m craving_mind --config config/default.yaml

# Продолжение прерванного прогона
python -m craving_mind --resume runs/run_<timestamp>/

# Наследование compress.py от предыдущего прогона
python -m craving_mind --config config/default.yaml --inherit runs/run_<timestamp>/
```

Дашборд запускается автоматически на `http://localhost:8080` если `dashboard.enabled: true` в конфиге.

---

## Ключевые концепции

### Экономика токенов

Единственная валюта существования агента — токены. Жёсткий лимит (`Hard Cap`) на эпоху (фиксированный батч из 10 задач). Превышение ведёт к `OOM_KILLED` и полному откату изменений. Бюджет единый для всего: рабочие задачи, рефлексия, чтение памяти, переписывание алгоритма.

- **Turn Budget Cap:** каждый вызов LLM ограничен 20% остатка бюджета. Максимум 3 round-trip'а за вызов.
- **Скупой Пульс:** единственная бесплатная телеметрия — строка `[B:14050|C:412]` (остаток бюджета | цена прошлого шага).
- **Минимум max_tokens:** если бюджет не позволяет выделить 200 токенов на ответ, вызов пропускается — OOM.

### Три фазы давления

Ограничения включаются последовательно. Агент не уведомляется о смене фазы.

| Фаза | Эпохи | Что добавляется |
|------|-------|-----------------|
| **Инкубатор** | 1–10 | Hard Cap, Turn Budget Cap, Скупой Пульс. Памяти нет, штрафов нет. `bible.md` недоступен. Экспоненциальный венчур. |
| **Становление** | 11–25 | `bible.md` и `graveyard.md`, DIY Routing, эндогенная компрессия памяти, R&D Slush Fund. |
| **Полное давление** | 26+ | Режим «Крыса» при критическом голодании, Протокол Шрамирования (LRU/TTL), Duplicate Filter. |

### Инструменты агента

Агент взаимодействует через 6 инструментов (phase-gated):

| Инструмент | Описание | Фаза |
|-----------|----------|------|
| `run_compress` | Запуск compress() на тестовых данных | 1+ |
| `read_file` | Чтение файлов workspace (compress.py, graveyard.md, bible.md) | 1+ (bible.md — 2+) |
| `write_file` | Полная перезапись файла (дорого) | 1+ (bible.md — 2+) |
| `edit_file` | Замена подстроки в compress.py (дёшево, max 500 chars old_string) | 1+ |
| `run_script` | Выполнение произвольного Python-скрипта в sandbox | 1+ |
| `audit_budget` | Проверка остатка бюджета и стоимости последней операции | 1+ |

### Провайдеры LLM

| Провайдер | Описание | Как использовать |
|-----------|----------|------------------|
| `cli` | Claude Code CLI SDK. Structured output через `--json-schema`. Не требует API-ключа. | `--provider cli` |
| `anthropic` | Anthropic Messages API. Function calling. Требует `ANTHROPIC_API_KEY`. | `--provider anthropic` |
| `mock` | Заглушка для тестов. | Только в тестах |

CLI провайдер: каждый `_run_turn()` начинает свежую сессию для надёжного structured output. Внутри turn'а tool loop работает через resumed session.

---

## Архитектура

```
┌─────────────────────────────────────────────────────┐
│                    АГЕНТ (LLM)                       │
│  compress.py  ←→  bible.md  ←→  graveyard.md        │
│  [единый бюджет токенов на всё]                      │
│  Инструменты: run_compress, read/write/edit_file,    │
│               run_script, audit_budget               │
└───────────────────────┬─────────────────────────────┘
                        │ метрики: ratio, sem, ent, pass/fail
                        │ (агент НЕ видит исходный текст)
                        ▼
┌─────────────────────────────────────────────────────┐
│                   СУДЬЯ (Judge)                      │
│  • Smoke test compress.py (gate)                     │
│  • Cosine similarity (MiniLM-L12-v2)                 │
│  • Entity F1 (spaCy en_core_web_lg)                  │
│  • Type-specific валидаторы (discourse/needle/code)  │
│  • CUSUM мониторинг дрейфа success_rate              │
└───────────────────────┬─────────────────────────────┘
                        │ {compression_ratio, semantic_score, entity_score, pass}
                        ▼
┌─────────────────────────────────────────────────────┐
│              ДАШБОРД ОПЕРАТОРА                       │
│  Live Console (с файлами) · Charts · Health ·        │
│  Budget · Artifacts · Phase Progress                 │
└─────────────────────────────────────────────────────┘
```

### Поток данных на задачу

1. Система автоматически запускает `compress.py` на source_text (агент не видит текст).
2. Судья оценивает результат: semantic score + entity F1.
3. Агент получает только 4 числа: `ratio`, `a` (semantic), `b` (entity), `PASS/FAIL`.
4. Агент реагирует: может читать/писать файлы, редактировать compress.py, запускать скрипты.

### Sandbox

Код агента выполняется в изолированном `subprocess` с ограничениями:
- **Таймаут:** 5 секунд
- **Allowed imports:** re, string, math, collections, itertools, functools, heapq, bisect, json, pathlib, io, textwrap, unicodedata, hashlib, difflib, struct
- **Запрещено:** os, subprocess, socket, requests, numpy, sklearn, spacy, nltk
- **Smoke test:** каждое обновление compress.py проверяется на 10 фиксированных текстах перед сохранением

---

## Дашборд

Real-time веб-интерфейс (FastAPI + WebSocket):
- **Live Console** — объединённая панель с вкладками: Console, compress.py, bible.md, graveyard.md, Artifact
- **Charts** — Success Rate, Semantic/Entity Score, Overfit Gap, Saved Tokens
- **Health** — Budget, Bible Weight, OOM Events, Starvation Rate
- **Controls** — Pause/Resume, Stop

Crav-сообщения в консоли сворачиваются (click-to-expand). Открытые плюсики сохраняются при автообновлении.

---

## Структура репозитория

```
CravingMind/
├── config/
│   └── default.yaml              # Единый конфиг
├── data/
│   ├── benchmarks/               # Замороженные Parquet-бенчмарки
│   └── sources/                  # 120 текстов (discourse/needle/code × 40)
├── docs/
│   ├── planning/                 # Архитектурные решения, план, структура
│   └── spec/
│       └── craving_mind_spec_v1_7_0.md
├── runs/                         # Артефакты экспериментов (не в git)
├── src/craving_mind/
│   ├── agent/                    # interface.py, tools.py, memory.py, sandbox.py
│   ├── benchmark/                # generator.py, loader.py, sources.py
│   ├── judge/                    # evaluator.py, embeddings.py, entities.py, scoring.py, ...
│   ├── orchestrator/             # runner.py, budget.py, phases.py, checkpoint.py, artifact_manager.py
│   ├── dashboard/                # server.py, metrics.py, storage.py
│   └── utils/                    # config.py, logging.py, tokens.py
├── tests/                        # 458 тестов (pytest)
├── pyproject.toml
└── README.md
```

---

## Текущий статус

Система полностью реализована и запущена в экспериментальном режиме. Все 8 фаз разработки завершены, 458 тестов проходят. Ведутся первые прогоны с CLI провайдером.
