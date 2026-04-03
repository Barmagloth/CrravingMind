# CravingMind — Открытые вопросы по реализации

## 1. Инфраструктура агента

**Q1.1. Какая LLM для агента?**
Спека фиксирует модель Judge'а (`claude-haiku-4-5-20251001`), но для самого агента — открытый параметр. Конфиг `default.yaml` ставит `claude-sonnet-4-6` как дефолт, есть поддержка CLI-провайдера с алиасом `haiku`. Варианты: Haiku (дёшево, быстро), Sonnet (умнее, дороже), Mock (тесты). От выбора зависит калибровка базового бюджета — у разных моделей разная «плотность мысли» на токен.

**Q1.2. Как считать токены?**
Спека говорит «input + output, суммарно». `utils/tokens.py` — существует, но нужно уточнить: Tiktoken (OpenAI)? Anthropic tokenizer? `len(text) // 4` как грубая оценка? Для Circuit Breaker нужен детерминированный подсчёт до отправки запроса.

**Q1.3. Sandbox для compress.py и DIY-скриптов**
Реализован в `agent/sandbox.py`. Текущий подход: `subprocess` + `resource limits` + allowlist импортов (numpy, sklearn, spacy, nltk, etc.), таймаут 5 сек. Docker-контейнер не используется (сложнее для кросс-платформы). Вопрос: достаточно ли subprocess-изоляции, или нужен более жёсткий sandboxing?

**Q1.4. Tool use vs. текстовые команды**
Текущая реализация в `agent/tools.py` — через tool use API (function calling). Вопрос: какой интерфейс для DIY-скриптов агента — отдельный tool `run_script(code)` или файловая операция `write_file` + `run_file`?

**Q1.5. Управление контекстным окном агента** ✅ РЕШЕНО
Свежая CLI-сессия на каждый `_run_turn()`. Внутри turn'а resumed session для tool loop (read → edit → run_compress). Между задачами conversation trimmed до последнего assistant summary (200 chars) + новые метрики. Агент восстанавливает контекст через `read_file`. Это даёт предсказуемые input tokens и надёжный structured output.

## 2. Judge / Scoring Pipeline

**Q2.1. Модель эмбеддингов для cosine similarity**
Решено: `sentence-transformers/all-MiniLM-L12-v2` (локально, через HuggingFace). Реализовано в `judge/embeddings.py`. Первый запуск требует скачивания модели (~120 MB).

**Q2.2. NER pipeline**
Конфиг: `en_core_web_sm` (дефолт), но spec упоминает `en_core_web_lg` как более точный. Реализован в `judge/entities.py`. Вопрос: переключаться ли на lg-модель для продакшена?

**Q2.3. Как Judge генерирует вопросы при сборке бенчмарка?**
LLM-вызов (Haiku) с `temperature=0, seed=42`. Формат: структурированный вывод (JSON-список из 10 вопросов). Реализован в `benchmark/generator.py`. Вопрос: нужен ли structured output (JSON mode) или достаточно prompt-инженерии?

**Q2.4. Порядок выполнения шагов Judge'а**
Текущая реализация в `judge/evaluator.py` — синхронный конвейер: ratio check → LLM questions → NER extraction → scoring. Вопрос: параллелизировать ли 10 вопросов?

## 3. Benchmark / Данные

**Q3.1. Источники исходных текстов**
Реализовано в `data/sources/`: `discourse/` (40 текстов с Wikipedia), `needle/` (40 текстов с Wikipedia), `code/` (40 реальных примеров кода). Итого 120 текстов для 3 типов. Вопрос: нужны ли дополнительные типы (legal, tables, dialogue)?

**Q3.2. Сколько задач в эпохе?**
Решено: `tasks_per_epoch: 10` (в конфиге). Рабочий пул = 70% бенчмарка, holdout = 30%.

**Q3.3. Гибридный бенчмарк (frozen + dynamic)**
Принято решение заменить чистый holdout на гибридную схему: 70% замороженный Parquet + 30% динамический (свежая выборка каждую эпоху). `dynamic_multiplier: 1.3` при агрегации success_rate. Реализовано в `benchmark/loader.py`.

**Q3.4. Валидация бенчмарка**
Обязательный ручной прогон перед заморозкой Parquet. Вопрос: какой минимальный размер бенчмарка для статистически значимых результатов?

## 4. Токен-экономика (калибровка)

**Q4.1. Базовый бюджет**
Решено: `base_tokens: 50000`. Обоснование: ~10 задач × ~4K токенов на задачу + ~10K на рефлексию и R&D.

**Q4.2. Lambda (λ) для R&D fund**
Решено: `rnd_lambda: 0.0001`. Формула: `R&D = 0.3 × base × (1 − e^{−λ × saved})`. При λ=0.0001 и сэкономленных 10K токенов даёт ~27% конверсию.

**Q4.3. TTL для graveyard.md**
Решено: `graveyard_ttl_epochs: 10`. Технические сбои (crash) — `immutable`, хранятся вечно.

**Q4.4. Venture decay коэффициент**
Решено: `venture_decay: 0.5`. Формула: `K = 1 + 2·e^{−0.5·epoch}`. К эпохе 10: K ≈ 1.01.

## 5. Dashboard оператора

**Q5.1. Новый или переиспользованный?**
Написан с нуля: `dashboard/server.py` (FastAPI + WebSocket), `dashboard/storage.py` (SQLite), `dashboard/metrics.py`. По умолчанию отключён (`enabled: false`), порт 8080.

**Q5.2. Storage для метрик**
SQLite через `dashboard/storage.py`. Вопрос: достаточно ли SQLite для long-running экспериментов (100+ эпох)?

**Q5.3. Что показывает дашборд**
Рабочий score vs. holdout score (расхождение = сигнал переобучения), CUSUM drift detection, budget utilization, фаза, артефакт версии compress.py.

## 6. Операционные вопросы

**Q6.1. Как запускать эксперимент?**
`python -m craving_mind --config config/default.yaml`. Точка входа: `__main__.py` → `orchestrator/runner.py`.

**Q6.2. Как останавливать и возобновлять?**
Checkpoint после каждой эпохи: `runs/run_<timestamp>/checkpoint.json`. Возобновление через `--resume runs/run_xxx/`.

**Q6.3. Версионирование конфигов**
Config immutable после старта прогона (копируется в `runs/run_xxx/`). Смена config = новый прогон.

**Q6.4. Inheritance между прогонами**
Флаг `inheritance.enabled: false` по умолчанию. При включении: `carry_graveyard: true`, `carry_compress: true`. Вопрос: как версионировать унаследованные артефакты?

## 7. Вопросы по архитектуре кода

**Q7.1. Монолит или микросервисы?**
Монолит: единый Python-пакет `src/craving_mind/` с логическими модулями. Оркестратор (`orchestrator/`) запускает Judge (`judge/`), агента (`agent/`), дашборд (`dashboard/`) в одном процессе.

**Q7.2. Async или sync?**
Синхронная основа с потенциалом для asyncio в dashboard (WebSocket). Вопрос: нужен ли async для параллельных вызовов Judge (10 вопросов → 10 LLM запросов)?

**Q7.3. Тестирование**
Покрыто юнит-тестами: scoring, dedup, entities, validators, sandbox, budget, benchmark. Интеграционные тесты: `test_integration.py`. Вопрос: нужны ли e2e тесты с реальным LLM, или достаточно Mock-провайдера?
