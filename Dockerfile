FROM python:3.10

WORKDIR /app/backend

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend

COPY Phase1 ./Phase1
COPY Phase2 ./Phase2
COPY Phase3 ./Phase3

# Cloud Run uses PORT env variable
ENV PORT=8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 120 AIController:app