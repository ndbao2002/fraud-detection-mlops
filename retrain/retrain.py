import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 1. Load prediction data with labels
df = pd.read_csv("output/predictions.csv")

# Make sure you have a 'prediction' column!
if 'prediction' not in df.columns:
    raise ValueError("Missing 'prediction' column in predictions.csv")
    
# 2. Extract features which is stored as string of list, add header
X = pd.DataFrame(df["features"].apply(lambda x: np.fromstring(x.strip("[]"), sep=',')).tolist(), 
                 columns=["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15",
                          "V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"])
y = df["prediction"]

# Pop Class column
X = X.drop(columns=['Class'])

# Handle unbalanced classes
smote = SMOTE(random_state=25)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 3. Scale only Time and Amount
scaler = StandardScaler()
X_resampled[['Time', 'Amount']] = scaler.fit_transform(X_resampled[['Time', 'Amount']])

# 4. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

# 5. Save model and scaler
joblib.dump(model, "models/random_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# 6. Print model accuracy
accuracy = model.score(X, y)
print(f"Model accuracy: {accuracy:.2f}")

# 7. Log via mlflow
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("credit_card_fraud_detection")

with mlflow.start_run():
    # Add model signature
    signature = mlflow.models.infer_signature(X.iloc[0:5], model.predict(X.iloc[0:5]))
    mlflow.sklearn.log_model(model, "random_forest_model", signature=signature, input_example=X.iloc[0:5])

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_artifact("models/random_forest_model.pkl")
    mlflow.log_artifact("models/scaler.pkl")
