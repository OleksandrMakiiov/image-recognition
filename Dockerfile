# Используем официальный Python образ в качестве основы
FROM python:3.12-slim

# Устанавливаем необходимые системные зависимости, включая Tesseract
RUN apt-get update && \
    apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Создаём рабочую директорию в контейнере
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY . /app/

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт для приложения (например, 8000 для FastAPI)
EXPOSE 8000

# Указываем команду для запуска FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
