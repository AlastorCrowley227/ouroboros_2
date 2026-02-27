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

Приложение **читает именно JSON-конфиг** через `load_runtime_config()` и затем экспортирует параметры в переменные окружения.

Обязательные поля зависят от `vcs_platform`:

- Всегда: `telegram_bot_token`
- Для `github`: `github_token`, `github_user`, `github_repo`
- Для `gitea`: `github_user`, `github_repo` + хотя бы одно из `gitea_base_url` или `git_remote_url`

### 2) Шаблон переменных окружения с комментариями

Файл: `.env.example`

Этот файл добавлен как удобная шпаргалка: в нём перечислены все ключевые переменные без значений и с пояснениями. Он **не заменяет** `ouroboros.config.json`, но помогает быстро понять, что и зачем нужно.

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

## Полезные команды

```bash
make test      # запуск тестов
make test-v    # подробный запуск тестов
make health    # метрики сложности/здоровья кода
```

## Примечания по безопасности

- Не коммитьте реальные токены/ключи в `ouroboros.config.json`.
- Используйте `.env.example` только как шаблон.
