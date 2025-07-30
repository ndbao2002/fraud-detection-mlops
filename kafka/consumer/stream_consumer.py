# stream_consumer.py
from datetime import datetime
import os
from kafka import KafkaConsumer
import json
import requests
import csv

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

csv_file = "data/prediction.csv"
write_header = not os.path.exists(csv_file)

consumer = KafkaConsumer(
    'fraud-data',
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    auto_offset_reset='earliest',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

with open(csv_file, mode="a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["timestamp", "features", "prediction"])

    # Consume messages from the Kafka topic
    for message in consumer:
        data = message.value
        features = list(data.values())

        # Send the data to the prediction API
        response = requests.post("http://fraud-api-service:8000/predict", json=data)
        
        if response.status_code == 200:
            prediction = response.json()
            pred_class = prediction.get('prediction')

            print(f"Prediction: {prediction}", flush=True)

            # Write the data to the CSV file
            writer.writerow([datetime.now().isoformat(), features, pred_class])
        else:
            print(f"Failed to get prediction: {response.status_code}, {response.text}", flush=True)

        f.flush()  # Ensure data is written to the file immediately
