FROM python:3.11-slim

# Arbeitsverzeichnis setzen
WORKDIR /app

# System-Pakete (z. B. für sqlite & SSL)
RUN apt-get update && apt-get install -y \
    sqlite3 libsqlite3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies
COPY requirements.txt .
RUN pip install pydantic-settings
RUN pip install --no-cache-dir -r requirements.txt

# App-Code reinkopieren
COPY . .

# Port freigeben
EXPOSE 8000

# Startbefehl → uvicorn mit FastAPI-App
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
