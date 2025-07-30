FROM python:3.10-slim

WORKDIR /app

COPY kafka/consumer/requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir data

COPY kafka/consumer/stream_consumer.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "stream_consumer.py"]