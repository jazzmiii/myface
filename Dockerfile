FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port
EXPOSE 7860

ENV FLASK_APP=app.py
ENV FLASK_RUN_PORT=7860
ENV FLASK_RUN_HOST=0.0.0.0

CMD ["flask", "run"]
