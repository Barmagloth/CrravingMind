# CravingMind — Структура проекта

## Корневая директория

```
R:/Projects/CravingMind/
├── config/
│   └── default.yaml              # Единый конфиг: бюджет, фазы, Judge, бенчмарк, дашборд
├── data/
│   ├── benchmarks/               # Замороженные Parquet-бенчмарки (генерируются, не в git)
│   └── sources/
│       ├── code/                 # 40 реальных примеров кода (алгоритмы, фреймворки, ML)
│       ├── discourse/            # 40 текстов с Wikipedia (дискурс, объяснения)
│       └── needle/               # 40 текстов с Wikipedia (факты, числа, имена)
├── docs/
│   ├── planning/                 # Планировочные документы (этот файл и другие)
│   └── spec/
│       ├── craving_mind_spec_v1_6_1.md  # Техническое задание (исходная версия)
│       └── craving_mind_spec_v1_7_0.md  # Техническое задание (текущая)
├── runs/                         # Артефакты экспериментов (не в git)
│   └── run_<timestamp>/
│       ├── checkpoint.json       # Состояние для возобновления
│       ├── epoch_log.jsonl       # Лог всех эпох (JSONL)
│       ├── craving_mind_<ts>.jsonl  # Полный event log
│       ├── agent_workspace/      # Файлы агента: bible.md, graveyard.md, compress.py
│       └── artifacts/            # Версионированные снимки compress.py
├── src/
│   └── craving_mind/             # Основной Python-пакет
├── tests/                        # Тесты (pytest)
├── .env.example                  # Шаблон env-переменных
├── .gitignore
├── pyproject.toml                # Зависимости и метаданные пакета
├── README.md                     # Обзор проекта, быстрый старт, архитектура
└── CLAUDE.md                     # Инструкции для AI-ассистентов (если создан)
```

---

## Основной пакет: `src/craving_mind/`

```
src/craving_mind/
├── __init__.py
├── __main__.py                   # Точка входа: python -m craving_mind
│
├── agent/                        # LLM-агент и его инфраструктура
│   ├── __init__.py
│   ├── interface.py              # AgentInterface + CLIProvider: LLM calls, structured output (--json-schema),
│   │                             #   fresh session per turn, tool loop с resumed session, скупой пульс [B:|C:]
│   ├── memory.py                 # Управление bible.md и graveyard.md (чтение/запись/валидация)
│   ├── sandbox.py                # Изолированный запуск compress.py и DIY-скриптов (subprocess + timeout)
│   └── tools.py                  # Tool definitions (phase-gated): run_compress, read_file, write_file,
│                                 #   edit_file (diff-only, max 500 chars), run_script, audit_budget
│
├── benchmark/                    # Сборка и загрузка бенчмарка
│   ├── __init__.py
│   ├── generator.py              # BenchmarkGenerator: генерация вопросов через LLM Haiku,
│                                 #   NER extraction, сохранение в Parquet
│   ├── loader.py                 # BenchmarkLoader: гибридный режим (frozen 70% + dynamic 30%),
│                                 #   ротация пулов, семплирование задач на эпоху
│   └── sources.py                # SourceManager: чтение текстов из data/sources/ по типу
│
├── judge/                        # Внешний детерминированный Judge
│   ├── __init__.py
│   ├── dedup.py                  # DedupFilter: SHA-256 хэш задач, детекция повторов
│   ├── drift.py                  # DriftDetector: CUSUM алгоритм, окно 10 эпох, порог 2σ
│   ├── embeddings.py             # EmbeddingModel: sentence-transformers/all-MiniLM-L12-v2,
│                                 #   cosine similarity для semantic score
│   ├── entities.py               # EntityExtractor: spaCy en_core_web_sm, NER для entity F1
│   ├── evaluator.py              # JudgeEvaluator: главный конвейер оценки задачи:
│                                 #   ratio check → LLM questions → NER → scoring → verdict
│   ├── scoring.py                # ScoringEngine: формулы task_score, epoch_score,
│                                 #   dynamic_multiplier, success_rate агрегация
│   ├── smoke_test.py             # SmokeTest: gate на исполняемость compress.py (10 фикс. текстов)
│   └── validators.py             # TypeValidators: type-specific проверки (code, discourse, needle)
│
├── orchestrator/                 # Оркестратор: управление экспериментом
│   ├── __init__.py
│   ├── artifact_manager.py       # ArtifactManager: версионирование compress.py,
│                                 #   архивирование после успешной эпохи
│   ├── budget.py                 # BudgetManager: Hard Cap, Circuit Breaker, Exponential Venture,
│                                 #   R&D Fund (carry-over), CRITICAL_STARVATION флаг
│   ├── checkpoint.py             # CheckpointManager: сохранение/загрузка состояния (JSON),
│                                 #   resume с произвольной эпохи
│   ├── phases.py                 # PhaseManager: автоматическое переключение фаз 1→2→3,
│                                 #   активация/деактивация механизмов по номеру эпохи
│   └── runner.py                 # ExperimentRunner: главный цикл эпох, интеграция всех модулей,
│                                 #   обработка OOM_KILLED, inheritance между прогонами
│
├── dashboard/                    # Дашборд оператора (real-time мониторинг)
│   ├── __init__.py
│   ├── metrics.py                # MetricsCollector: сбор и агрегация метрик эпохи
│   ├── server.py                 # DashboardServer: FastAPI + WebSocket, HTTP endpoints,
│   │                             #   Live Console с файловыми вкладками, crav toggle preservation,
│   │                             #   Charts (SR, Scores, Overfit, Tokens), Health, Controls
│   └── storage.py                # MetricsStorage: SQLite, запись/чтение истории эпох
│
└── utils/                        # Общие утилиты
    ├── __init__.py
    ├── config.py                 # ConfigLoader: загрузка YAML, валидация, доступ через dataclass
    ├── logging.py                # Structured logging: JSONL event log, human-readable консоль
    └── tokens.py                 # TokenCounter: подсчёт токенов (approximation: len//4 или tiktoken)
```

---

## Тесты: `tests/`

```
tests/
├── __init__.py
├── fixtures/
│   └── sample_texts/            # Короткие тексты для unit тестов
│       ├── discourse/
│       ├── needle/
│       └── empty_type/
├── test_agent.py                # AgentInterface, memory management, tool use
├── test_artifact_manager.py     # Версионирование, архивирование compress.py
├── test_benchmark.py            # BenchmarkGenerator, BenchmarkLoader, гибридный режим
├── test_budget.py               # BudgetManager: все формулы (venture, R&D, circuit breaker)
├── test_dashboard.py            # MetricsStorage, MetricsCollector, server endpoints
├── test_dedup.py                # DedupFilter: хэши, детекция дублей
├── test_entities.py             # EntityExtractor: NER, entity F1 расчёт
├── test_integration.py          # End-to-end: полный прогон одной эпохи с Mock LLM
├── test_orchestrator.py         # ExperimentRunner, фазы, OOM handling
├── test_phases.py               # PhaseManager: переключение фаз, активация механизмов
├── test_sandbox.py              # Sandbox: timeout, allowlist, error handling
├── test_scoring.py              # ScoringEngine: формулы, пороги, агрегация
└── test_validators.py           # TypeValidators: type-specific проверки
```

---

## Конфиг: `config/default.yaml`

Единственный конфиг-файл. Покрывает:
- **agent**: провайдер (anthropic/cli/mock), модель, cli_model
- **phases**: границы фаз (11, 26)
- **budget**: base_tokens=5000, circuit_breaker=15%, rnd_lambda=0.0001
- **memory**: bible max_weight=20%
- **sandbox**: timeout=5s, allowed_imports (stdlib only: re, math, collections, json, hashlib, difflib и др.)
- **benchmark**: frozen_ratio=0.7, tasks_per_epoch=10, n_questions=10, dynamic_ratio=0.3
- **judge**: pass_threshold=0.85, dynamic_multiplier=1.3, embeddings model, NER model, LLM model
- **dashboard**: enabled=false, port=8080
- **inheritance**: enabled=false, carry_graveyard, carry_compress
- **logging**: level=INFO

Конфиг копируется в `runs/run_<timestamp>/` при старте и становится immutable для данного прогона.

---

## Директория прогона: `runs/run_<timestamp>/`

```
runs/run_20260330T220439Z/
├── checkpoint.json              # {"epoch": 5, "phase": 1, "budget_remaining": 42000, ...}
├── epoch_log.jsonl              # Одна JSON-строка на эпоху: метрики, verdict, budget stats
├── craving_mind_20260330T220439Z.jsonl  # Полный event log (все tool calls, LLM responses)
├── agent_workspace/
│   ├── compress.py              # Текущая версия функции сжатия
│   ├── bible.md                 # Правила и эвристики агента (Фаза 2+)
│   └── graveyard.md             # История ошибок (Фаза 2+)
└── artifacts/
    ├── compress_v1_epoch3.py    # Снимок после успешной эпохи 3
    ├── compress_v2_epoch7.py    # ...
    └── compress_latest.py       # Симлинк/копия лучшей версии
```
