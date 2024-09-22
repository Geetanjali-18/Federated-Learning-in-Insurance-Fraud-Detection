import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
import flwr as fl
import task2

if __name__ == "__main__":
    N_CLIENTS = 10

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()
    partition_id = args.partition_id

    # Load your CSV file
    df = pd.read_csv("new_dataset.csv")  # Replace with your CSV file path

    # Assuming your CSV has features and a label column named 'fraud_reported'
    X = df.drop(columns=["fraud_reported"], axis=1).values  # Adjust column name if needed
    y = df["fraud_reported"].values

    # Split the dataset into training and test sets
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]

    # Partition the dataset for the specific client
    client_size = len(X_train) // N_CLIENTS
    start = partition_id * client_size
    end = start + client_size if partition_id < N_CLIENTS - 1 else len(X_train)
    
    X_train_client = X_train[start:end]
    y_train_client = y_train[start:end]

    # Create a DecisionTreeClassifier model
    model = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        random_state=42,
    )
    model.fit(X_train_client, y_train_client) 
    task2.set_initial_params(model)  # Call your initial parameters function

    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return task2.get_model_parameters(model)

        def fit(self, parameters, config):
            task2.set_model_params(model, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_client, y_train_client)  # Use client-specific data
            print(f"Training finished for round {config['server_round']}")
            return task2.get_model_parameters(model), len(X_train_client), {}

        def evaluate(self, parameters, config):
            task2.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    def client_fn():
        numpy_client = FlowerClient()  # Create an instance of your NumPyClient
        return numpy_client.to_client()  # Convert it to Client

    # Start the client
    fl.client.start_client(server_address="127.0.0.1:8080", client=client_fn())
