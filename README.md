# 📰 Финансовый суммаризатор новостей

## О проекте
Небольшое Streamlit-приложение для получения summary финансовых новостей.
Под капотом используется llama-cpp-python с локальными GGUF-моделями (по умолчанию Qwen), а при их отсутствии включается простой fallback-алгоритм.

## Основные возможности
- Ввод произвольного текста статьи и получение краткой выжимки.
- Подключение любых локальных LLM-моделей в формате `.gguf` без обращения к внешним API.
- Пошаговая потоковая генерация вывода (stream) для плавного отображения результата.
- Готовые скрипты для: загрузки новостей Yahoo Finance, генерации датасетов и бенчмаркинга моделей.
- Docker-образ с полным окружением (Ubuntu 22.04 + Python 3.11).

## Быстрый запуск
### Docker Compose (рекомендуется)
```bash
mkdir -p data/models            # каталог для моделей
# поместите *.gguf файлы внутрь data/models/
docker-compose up -d            # сборка и запуск
```
Приложение будет доступно на `http://localhost:8501`.

P.S: Инференс на streamlit занимает какое-то время даже со streaming (около 2-3 минут). Пока не успел это пофиксить. Если хочется быстрого результата заходим в контейнер и запускаем scripts/summarize_news.py

### Ручной запуск в Docker (пока CPU version only)
```bash
docker build -t news-summarizer .
docker run -d -p 8501:8501 \
  -v /path/to/gguf:/app/data/models \
  news-summarizer

docker exec -ti <container_id> bash
python3 scripts/summarize_news.py
```

### Запуск без контейнера (позволяет использовать MLX на Apple Silicon)
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Структура репозитория
```text
financial-news-summarizer/
├── app.py                   # Web-интерфейс Streamlit
├── prompts.py               # Системный и few-shot промпты
├── scripts/                 # Утилиты CLI
│   ├── load_news.py         # Загрузка RSS и парсинг статей
│   ├── summarize_news.py    # Обёртка QwenModel (llama-cpp)
│   ├── process_dataset.py   # Генерация датасета локальной моделью
│   └── process_dataset_gpt.py # То же через OpenAI/OpenRouter
├── data/
│   └── models/              # *.gguf модели (монтируется в конт)
├── Dockerfile               # Cборка окружения
├── docker-compose.yml       # Быстрый запуск
├── requirements.txt         # Зависимости рантайма
└── README.md
```

## Описание компонентов
- **`app.py`** – форма ввода текста, выбор модели, потоковая генерация результата.
- **`QwenModel`** (`scripts/summarize_news.py`) – класс-обёртка над llama-cpp с удобными методами `run`, `run_stream`, `count_tokens`.
- **Скрипты** в `scripts/` автоматизируют сбор новостей и подготовку обучающих/валид-датасетов.
- **`prompts.py`** хранит строгий system-prompt и few-shot примеры для финансового домена.
- **Dockerfile** собирает минимальный образ; переменные окружения задают оптимизационные флаги для компиляции GGML.

## Работа с моделями
1. Поместите `.gguf` файл в `data/models/` (или смонтируйте через `docker-compose`).
2. Приложение автоматически найдёт его и предложит в выпадающем списке.
3. Параметры генерации (контекст 16k, temperature, top-p и т. д.) задаются в `scripts/summarize_news.py`.

## Зависимости
- Runtime: `requests`, `pandas`, `feedparser`, `newspaper4k`, `tqdm`, `streamlit`, `transformers`, `torch`, `llama-cpp-python` (устанавливается отдельно), `gguf`.
- Dev: `black`, `ruff`, `pytest`, `pre-commit`.
