# CravingMind — План реализации (Фазы 0–8)

## Обзор

9 фаз от нулевой инфраструктуры до полного e2e эксперимента. Фазы 0–8 соответствуют фактическому ходу разработки.

---

## Фаза 0 — Основа проекта

**Что делали:** Инициализация Python-пакета, структура директорий, базовый конфиг.

**Результат:**
- `src/craving_mind/` пакет с `pyproject.toml`
- `config/default.yaml` — единый конфиг всех параметров
- `utils/config.py` — загрузка конфига
- `utils/logging.py` — structured logging (JSONL + консоль)
- `.env.example`, `.gitignore`

**Зависимости:** нет (точка старта)

---

## Фаза 1 — Judge: Scoring Pipeline

**Что делали:** Внешний детерминированный конвейер оценки качества сжатия.

**Результат:**
- `judge/embeddings.py` — cosine similarity (MiniLM-L12-v2)
- `judge/entities.py` — NER extraction (spaCy)
- `judge/scoring.py` — task_score, epoch_score, success_rate агрегация
- `judge/validators.py` — type-specific валидаторы
- `judge/dedup.py` — SHA-256 дедупликация задач
- `judge/evaluator.py` — полный конвейер: ratio → LLM → NER → score → verdict
- `judge/smoke_test.py` — gate на исполняемость compress.py
- Тесты: `test_scoring.py`, `test_entities.py`, `test_dedup.py`, `test_validators.py`

**Зависимости:** Фаза 0

---

## Фаза 2 — Benchmark: Сборка и загрузка

**Что делали:** Генератор бенчмарка + гибридный loader.

**Результат:**
- `benchmark/sources.py` — чтение текстов из `data/sources/` по типу
- `benchmark/generator.py` — LLM-генерация вопросов, NER, сохранение Parquet
- `benchmark/loader.py` — гибридный режим (70% frozen + 30% dynamic), семплирование
- Данные: 40 текстов × 3 типа (discourse, needle, code) в `data/sources/`
- Тест: `test_benchmark.py`

**Зависимости:** Фаза 1 (Judge нужен для генерации вопросов)

---

## Фаза 3 — Orchestrator: Бюджет и фазы давления

**Что делали:** Управление токен-экономикой, переключение фаз, checkpoint.

**Результат:**
- `orchestrator/budget.py` — Hard Cap, Circuit Breaker, Exponential Venture, R&D Fund, CRITICAL_STARVATION
- `orchestrator/phases.py` — автопереключение Phase 1→2→3 по номеру эпохи
- `orchestrator/checkpoint.py` — сохранение/загрузка состояния (JSON), resume
- `orchestrator/artifact_manager.py` — версионирование compress.py
- Тесты: `test_budget.py`, `test_phases.py`, `test_artifact_manager.py`

**Зависимости:** Фаза 0

---

## Фаза 4 — Agent: Инфраструктура агента

**Что делали:** LLM-агент с tool use, память, sandbox для скриптов.

**Результат:**
- `agent/interface.py` — AgentInterface: LLM calls, системный промпт со скупым пульсом `[B:|C:]`
- `agent/tools.py` — tool definitions: `run_compress`, `write_bible`, `read_bible`, `run_script`, `audit_budget`, `compact_bible`
- `agent/memory.py` — bible.md и graveyard.md: AMENDMENT блоки, TTL/LRU для graveyard
- `agent/sandbox.py` — subprocess sandbox: timeout, allowlist импортов, error reporting
- CLIProvider: запуск через Claude CLI без API-ключа
- Тесты: `test_agent.py`, `test_sandbox.py`

**Зависимости:** Фазы 0, 3

---

## Фаза 5 — Runner: Главный цикл эпох

**Что делали:** Интеграция всех модулей в единый цикл эксперимента.

**Результат:**
- `orchestrator/runner.py` — ExperimentRunner: цикл эпох, координация Agent↔Judge↔Benchmark↔Budget
- `__main__.py` — CLI точка входа (`python -m craving_mind --config ... [--resume ...]`)
- Обработка OOM_KILLED (откат bible.md до начала эпохи)
- Inheritance между прогонами (carry_compress, carry_graveyard)
- Тест: `test_orchestrator.py`

**Зависимости:** Фазы 1–4

---

## Фаза 6 — Drift Detection

**Что делали:** CUSUM мониторинг дрейфа success_rate.

**Результат:**
- `judge/drift.py` — DriftDetector: CUSUM, скользящее окно 10 эпох, порог 2σ, флаг DATA_DRIFT
- Интеграция в Runner: drift check после каждой эпохи
- Тест (в рамках judge тестов)

**Зависимости:** Фаза 5

---

## Фаза 7 — Dashboard: Мониторинг оператора

**Что делали:** Real-time веб-интерфейс для наблюдения за экспериментом.

**Результат:**
- `dashboard/storage.py` — SQLite: запись/чтение истории эпох
- `dashboard/metrics.py` — сбор метрик: working score vs. dynamic score, budget, фаза, drift
- `dashboard/server.py` — FastAPI + WebSocket: HTTP API + push обновления каждые 2 сек
- Интеграция в Runner: dashboard обновляется по событиям
- Тест: `test_dashboard.py`

**Зависимости:** Фаза 5

---

## Фаза 8 — Integration Tests & Artifact Versioning

**Что делали:** End-to-end тесты, финальная валидация системы.

**Результат:**
- `test_integration.py` — полный прогон одной эпохи с Mock LLM провайдером
- Проверка: бюджетирование, scoring, checkpoint, artifact versioning
- Финальный runs layout: `checkpoint.json`, `epoch_log.jsonl`, `agent_workspace/`, `artifacts/`
- Исправление e2e багов, выявленных интеграционными тестами

**Зависимости:** Фазы 1–7

---

## Текущий статус (на 2026-03-31)

| Фаза | Статус | Ключевые файлы |
|------|--------|----------------|
| 0 — Основа | ✅ Готово | `config/`, `utils/` |
| 1 — Judge | ✅ Готово | `judge/` (7 модулей) |
| 2 — Benchmark | ✅ Готово | `benchmark/` (3 модуля) + 120 текстов |
| 3 — Orchestrator | ✅ Готово | `orchestrator/` (budget, phases, checkpoint, artifacts) |
| 4 — Agent | ✅ Готово | `agent/` (4 модуля) + CLIProvider |
| 5 — Runner | ✅ Готово | `orchestrator/runner.py`, `__main__.py` |
| 6 — Drift | ✅ Готово | `judge/drift.py` |
| 7 — Dashboard | ✅ Готово | `dashboard/` (3 модуля) |
| 8 — Integration | ✅ Готово | `test_integration.py`, e2e validated |

**Следующие шаги:**
1. Первый реальный прогон эксперимента (не Mock): `python -m craving_mind --config config/default.yaml`
2. Ручная калибровка: запустить 10–20 эпох, проверить scoring на репрезентативность
3. Принять решение по Q1.1 (модель агента: Haiku vs Sonnet) на основе первых данных
4. Документация: CLAUDE.md для новых контрибьюторов

---

## Ключевые архитектурные инварианты

1. **Judge детерминирован** — никаких случайных значений. Воспроизводимость результатов обязательна.
2. **Агент не знает о Judge** — агент получает только числа (score, pass/fail). Никаких подсказок.
3. **Бюджет общий** — R&D, рефлексия, задачи — всё из одного пула.
4. **Конфиг immutable** после старта прогона — для воспроизводимости экспериментов.
5. **Оператор наблюдает, не вмешивается** — автономная система. Ручное вмешательство только при DATA_DRIFT или аномалии.
