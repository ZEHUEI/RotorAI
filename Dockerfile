FROM python:3.10

WORKDIR /app

COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend .

COPY Phase1 ./Phase1
COPY Phase2 ./Phase2
COPY Phase3 ./Phase3

# Cloud Run uses PORT env variable
ENV PORT=8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 AIController:app