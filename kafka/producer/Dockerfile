FROM python:3.10-slim

WORKDIR /app

COPY kafka/producer/requirements.txt .
RUN pip install -r requirements.txt

COPY data/test.csv data/

COPY kafka/producer/stream_producer.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "stream_producer.py"]