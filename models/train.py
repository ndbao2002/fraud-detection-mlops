import joblib
import mlflow
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def train_model(X_train, y_train):
    """
    Train a Random Forest model using the provided training data.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.

    Returns:
    RandomForestClassifier: The trained model.
    """
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=4)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Scale features Time and Amount at the same time
    scaler.fit(X_resampled[['Time', 'Amount']])
    X_resampled[['Time', 'Amount']] = scaler.transform(X_resampled[['Time', 'Amount']])

    # Add mlflow tracking, make it observable at 127.0.0.1:8080
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("credit_card_fraud_detection")

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_resampled, y_resampled)

        # Add model signature
        signature = mlflow.models.infer_signature(X_resampled.iloc[0:5], model.predict(X_resampled.iloc[0:5]))

        # Log the model with MLflow
        mlflow.sklearn.log_model(model, "random_forest_model", signature=signature, input_example=X_resampled.iloc[0:5])

        return model

def evaluate_model(model: RandomForestClassifier, X_test, y_test):
    """
    Evaluate the trained model on the test data.

    Parameters:
    model (RandomForestClassifier): The trained model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test labels.

    Returns:
    float: The accuracy of the model on the test set.
    """
    # Scale test features
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    return accuracy

def main(X_train, y_train, X_test, y_test):
    """
    Main function to train and evaluate the model.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test labels.
    """
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    
    print(f"Model accuracy: {accuracy:.2f}")

    return model

if __name__ == "__main__":

    # Load sample data
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    X_train = train.drop(columns=['Class'])
    y_train = train['Class']
    X_test = test.drop(columns=['Class'])
    y_test = test['Class']

    model = main(X_train, y_train, X_test, y_test)

    # Save the model and scaler
    joblib.dump(model, 'models/random_forest_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')