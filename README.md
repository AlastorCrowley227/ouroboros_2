# Ouroboros

**Текущая версия: 6.2.0**

Ouroboros — локальный self-hosted агент с Telegram-интерфейсом, который работает в git-репозитории, умеет запускать инструменты (shell/git/web/browser/vision), вести память и проходить циклы эволюции.

- Точка входа: `launcher.py`
- Конфиг рантайма: `ouroboros.config.json`
- Основная логика: `supervisor/` + `ouroboros/`

## Что есть в проекте сейчас

- **Supervisor-слой**: очередь задач, воркеры, маршрутизация входящих сообщений, события, git-операции, Telegram-клиент.
- **Agent-слой**: цикл выполнения инструментов, контекст, память, LLM-обвязка, review/метрики, фоновые процессы.
- **Набор инструментов** (`ouroboros/tools`): `core`, `git`, `github`, `shell`, `search`, `browser`, `vision`, `knowledge`, `review`, `control`, `health`, `compact_context`, `evolution_stats` и др.
- **Тесты**: smoke/unit тесты в `tests/`.
- **Статическая документация**: `docs/`.

## Требования

- Python **3.10+**
- Linux/macOS окружение с доступом к git
- Telegram bot token

## Быстрый старт

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Скопируйте шаблон конфига и заполните значения:

```bash
cp ouroboros.config.example.json ouroboros.config.json
```

3. Запустите рантайм:

```bash
python launcher.py
```

## Конфигурация

### 1) Основной конфиг рантайма (обязательный)

Файл: `ouroboros.config.json`

Приложение **читает именно JSON-конфиг** через `load_runtime_config()` и затем экспортирует параметры в переменные окружения (включая `OLLAMA_BASE_URL`/`OLLAMA_API_KEY` для LLM-клиента).

Обязательные поля зависят от `vcs_platform`:

- Всегда: `telegram_bot_token`
- Для `github`: `github_token`, `github_user`, `github_repo`
- Для `gitea`: `github_user`, `github_repo` + хотя бы одно из `gitea_base_url` или `git_remote_url`

### 2) Шаблон переменных окружения с комментариями

Файл: `.env.example`

Этот файл — зеркальный шаблон env-переменных к `ouroboros.config.json`: те же ключи в формате окружения. Он **не заменяет** `ouroboros.config.json`, но нужен как справочник для деплоя/интеграций, где удобнее задавать env напрямую.

## Структура проекта

```text
.
├── launcher.py
├── ouroboros/
│   ├── agent.py
│   ├── config.py
│   ├── loop.py
│   ├── llm.py
│   ├── memory.py
│   ├── review.py
│   └── tools/
├── supervisor/
│   ├── workers.py
│   ├── queue.py
│   ├── telegram.py
│   ├── git_ops.py
│   ├── state.py
│   └── events.py
├── tests/
├── docs/
└── ouroboros.config.example.json
```


## Рекомендации по Ollama и большому контексту

Чтобы агент стабильно работал с tools и длинной историей (16k–32k), рекомендуется создать отдельную модель с зафиксированным `num_ctx` и использовать её как `OUROBOROS_MODEL`:

```bash
cat > Modelfile.32k <<'EOF'
FROM llama3.2:latest
PARAMETER num_ctx 32768
EOF

ollama create llama3.2-32k:latest -f Modelfile.32k
export OUROBOROS_MODEL=llama3.2-32k:latest
export OLLAMA_NUM_CTX=32768
```

LLM-клиент Ouroboros делает idempotent warmup через `/api/chat` перед первым рабочим вызовом каждой модели, чтобы загрузить модель с нужным контекстом. По умолчанию используется стратегия `OLLAMA_ENDPOINT_STRATEGY=single_v1` (единый endpoint `/v1/chat/completions` для всех запросов после warmup), что снижает риск случайной перезагрузки модели с меньшим контекстом.

## Полезные команды

```bash
make test      # запуск тестов
make test-v    # подробный запуск тестов
make health    # метрики сложности/здоровья кода
```

## Примечания по безопасности

- Не коммитьте реальные токены/ключи в `ouroboros.config.json`.
- Используйте `.env.example` только как шаблон.
