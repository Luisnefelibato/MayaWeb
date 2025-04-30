FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias para PyAudio
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponemos el puerto que usará la aplicación
EXPOSE $PORT

# Comando para iniciar la aplicación con Gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT app:app