import argparse
import os
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import hypertune


def load_data_from_bigquery(bigquery_table_path):
    """Load data from a BigQuery table."""
    from google.cloud import bigquery
    client = bigquery.Client()
    query = f"SELECT * FROM `{bigquery_table_path}`"
    dataframe = client.query(query).to_dataframe()
    return dataframe


def main(args):
    # Step 1: Load data
    print("Loading data...")
    if args.data_source_bigquery_table_path:
        data = load_data_from_bigquery(args.data_source_bigquery_table_path)
    else:
        raise ValueError("No data source provided.")
    
    print(f"Data loaded. Shape: {data.shape}")

    # Step 2: Prepare features and labels
    if args.target_column not in data.columns:
        raise ValueError(f"Target column '{args.target_column}' not found in data.")
    X = data.drop(columns=[args.target_column])
    y = data[args.target_column]
    print(f"Target column: {args.target_column}")

    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train XGBoost model
    print("Training model...")
    model = xgb.XGBClassifier(
        learning_rate=args.learning_rate,
        n_estimators=100,
        max_depth=6,
        objective="binary:logistic",
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # Step 4: Evaluate the model
    print("Evaluating model...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    logloss = log_loss(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Log Loss: {logloss}")
    print(f"Accuracy: {accuracy}")

    # Step 5: Report metrics to Vertex AI Hyperparameter Tuning
    print("Reporting metrics to Vertex AI...")
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="logloss",
        metric_value=logloss,
        global_step=0
    )

    # Step 6: Save the trained model
    print("Saving model...")
    os.makedirs(args.model_output_path, exist_ok=True)
    model_file = os.path.join(args.model_output_path, "model.bst")
    model.save_model(model_file)
    print(f"Model saved to {model_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an XGBoost model for binary classification.")

    # Input arguments
    parser.add_argument("--data_source_bigquery_table_path", type=str, required=True,
                        help="Path to the BigQuery table containing training data.")
    parser.add_argument("--target_column", type=str, required=True,
                        help="Name of the target column.")
    parser.add_argument("--learning_rate", type=float, required=True,
                        help="Learning rate for the XGBoost model.")
    
    # Output arguments
    parser.add_argument("--model_output_path", type=str, required=True,
                        help="Path to save the trained model.")
    
    args = parser.parse_args()
    main(args)