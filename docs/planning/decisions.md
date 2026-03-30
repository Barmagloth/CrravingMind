# CravingMind — Принятые решения

## Изменения относительно спеки v1.6.1

| # | Область | Решение в спеке v1.6.1 | Принятое решение | Обоснование |
|---|---------|----------------------|-----------------|-------------|
| 1 | Бенчмарк | Чистый holdout (70/30 frozen split) | **Гибридный бенчмарк**: 70% замороженный Parquet + 30% динамический пул (свежая выборка каждую эпоху) | Holdout в spec решает проблему переобучения на рабочем пуле, но не защищает от memorization заморозки в целом. Динамический пул создаёт постоянное давление «новизны». |
| 2 | Агрегация success_rate | Равные веса frozen/holdout | **`dynamic_multiplier: 1.3`** — динамический пул весит на 30% больше при агрегации | Более свежие данные должны сильнее влиять на решения о carry-over и фазах. |
| 3 | Роль оператора | «Наблюдающий оператор» — пассивный мониторинг | **Автономная система** с человеком только в петле мониторинга (dashboard). Оператор не вмешивается в эпохи. | Ручное вмешательство разрушает эволюционное давление. Operator action = только при DATA_DRIFT или аномалии. |
| 4 | Inheritance | Не описано в spec | **Реализован механизм наследования** (`inheritance.enabled`): перенос `graveyard.md` и `compress.py` между прогонами | Позволяет продолжить эволюцию без cold start при смене конфига. |
| 5 | NER модель | `en_core_web_lg` в описании | **`en_core_web_sm`** в `default.yaml` | MVP-решение: sm быстрее, достаточно для entity F1 на English текстах. Переключение на lg — конфиговый параметр. |

---

## Таблица реализационных решений

### Инфраструктура агента

| Параметр | Решение | Файл | Примечания |
|----------|---------|------|------------|
| LLM провайдер агента | `anthropic` (API) | `config/default.yaml` | Альтернатива: `cli` (без API-ключа), `mock` (тесты) |
| Модель агента (default) | `claude-sonnet-4-6` | `config/default.yaml` | Дороже Haiku, умнее для R&D фазы |
| CLI alias | `haiku` | `config/default.yaml` | Для запуска через CLI-провайдер |
| Tool use интерфейс | Function calling (Anthropic tool use API) | `agent/tools.py` | Структурированные вызовы: `run_compress`, `write_bible`, `read_bible`, `run_script`, `audit_budget` |
| Контекст между задачами | Stateless (новый вызов на каждую задачу) | `agent/interface.py` | Бюджет и статус передаются через системный промпт `[B:xxx|C:xxx]` |
| Sandbox реализация | `subprocess` + allowlist импортов + таймаут 5 сек | `agent/sandbox.py` | Docker не используется (сложнее для Windows). Allowlist: numpy, sklearn, spacy, nltk, re, math, collections, etc. |

### Judge / Scoring

| Параметр | Решение | Файл | Примечания |
|----------|---------|------|------------|
| Модель Judge (LLM) | `claude-haiku-4-5-20251001` | `config/default.yaml` | Фиксированный snapshot. temperature=0, seed=42. |
| Эмбеддинги | `sentence-transformers/all-MiniLM-L12-v2` | `judge/embeddings.py` | Локально, ~120 MB. Cosine similarity для semantic score. |
| NER pipeline | spaCy `en_core_web_sm` | `judge/entities.py` | Entity F1 для factual score. Веса: semantic=0.5, entity=0.5. |
| Pass threshold | `0.85` | `config/default.yaml` | Минимальный score для `PASS`. |
| Ratio tolerance | `1.05` | `config/default.yaml` | len(compressed)/len(original) ≤ target_ratio × 1.05. |
| Вопросов на задачу | `10` | `config/default.yaml` | Покрывают общий смысл + детали (числа, имена, логика). |
| Drift detection | CUSUM, окно 10 эпох, порог 2σ | `judge/drift.py` | Флаг `DATA_DRIFT` в дашборде, без автоматических действий. |
| Dedup хэш | SHA-256(lowercase(strip(text[:500])) + str(ratio)) | `judge/dedup.py` | Дешёвая эвристика. Prefix length конфигурируемый (500). |

### Токен-экономика

| Параметр | Значение | Формула / Обоснование |
|----------|----------|-----------------------|
| Базовый бюджет | `50 000` токенов | ~10 задач × 4K + 10K на рефлексию/R&D |
| Circuit Breaker | `15%` от эффективного бюджета | Не более 7 500 токенов на одну задачу |
| Venture decay | `0.5` | K = 1 + 2·e^{−0.5·epoch}. К эпохе 10: K ≈ 1.01 |
| R&D lambda | `0.0001` | R&D = 0.3 × base × (1 − e^{−λ × saved}). При 10K saved → ~27% |
| R&D max | `30%` от base | Асимптота формулы. |
| R&D условие | success_rate ≥ 50% + нет OOM_KILLED | Carry-over только при успешной эпохе |
| Critical starvation | `10%` остатка бюджета | Включает режим «Крыса» (Фаза 3) |
| graveyard TTL | `10` эпох | Логические ошибки. Технические сбои — immutable. |
| bible max weight | `20%` бюджета | При превышении — обязательная эндогенная компрессия |

### Бенчмарк

| Параметр | Решение | Примечания |
|----------|---------|------------|
| Структура | Гибрид: 70% frozen Parquet + 30% динамический | `benchmark/loader.py` |
| Задач в эпохе | `10` | `tasks_per_epoch` в конфиге |
| Источники | 3 типа: discourse, needle, code | По 40 текстов каждый в `data/sources/` |
| target_ratio range | `[0.2, 0.6]` | Равномерное распределение при генерации |
| Схема Parquet | `{source_text, hidden_type, questions[10], reference_answers[10], reference_entities[10]}` | hidden_type скрыт от агента |
| Вопросы | Генерируются LLM Haiku один раз при сборке, замораживаются | Deterministic: temperature=0, seed=42 |
| Ротация holdout | Раз в N эпох (конфигурируемо): 10% пулов меняются местами | Агент не знает о ротации |

### Фазы давления

| Фаза | Эпохи | Что активируется |
|------|-------|-----------------|
| Фаза 1 — Инкубатор | 1–10 | Hard Cap, Exponential Venture, Circuit Breaker, Скупой Пульс |
| Фаза 2 — Становление | 11–25 | + bible.md, graveyard.md, DIY Routing, Эндогенная компрессия, R&D Fund |
| Фаза 3 — Полное давление | 26+ | + Режим «Крыса», Протокол Шрамирования (TTL/LRU), Duplicate Filter |

### Дашборд

| Параметр | Решение |
|----------|---------|
| Фреймворк | FastAPI + WebSocket (real-time) |
| Storage | SQLite (`dashboard/storage.py`) |
| Порт | `8080` |
| Default | Отключён (`enabled: false`) |
| Метрики | Working score vs. dynamic score, budget utilization, фаза, drift status, compress.py версия |

### Архитектура кода

| Решение | Обоснование |
|---------|-------------|
| Монолит (один пакет) | Проще для research-проекта. Логические модули: `agent/`, `judge/`, `orchestrator/`, `benchmark/`, `dashboard/`, `utils/` |
| Sync Python | Основа синхронная. Dashboard использует asyncio для WebSocket. |
| `src/` layout | Стандарт для installable Python packages (`pip install -e .`) |
| Config через YAML | Единый `config/default.yaml`. Immutable после старта прогона. |
| Runs layout | `runs/run_<timestamp>/`: checkpoint, artifacts, agent_workspace, epoch_log |
| Тесты | pytest. Отдельные тесты на каждый модуль. Mock LLM провайдер для unit тестов. |
