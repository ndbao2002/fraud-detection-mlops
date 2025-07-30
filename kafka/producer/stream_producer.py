# stream_producer.py
import os
from kafka import KafkaProducer
import pandas as pd
import json
import time

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

df = pd.read_csv("data/test.csv")  # Your processed test data

# Randomly select a row to simulate real-time data streaming
while True:
    row = df.sample(n=1).to_dict(orient='records')[0]
    producer.send('fraud-data', value=row)
    print(f"Sent data: {row}", flush=True)
    time.sleep(10)  # Sleep for 10 seconds before sending the next record
