import flwr as fl
import task2
from flwr.common import NDArrays, Scalar
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, Optional, Tuple
from typing import Dict
import pandas as pd

# Load your CSV dataset once (adjust the path and structure as needed)
df = pd.read_csv("new_dataset.csv")  # Replace with your CSV file path
X = df.drop(columns=["fraud_reported"]).values  # Drop the label column to get features
y = df["fraud_reported"].values  # Extract labels

# Split the dataset into test data
X_test, y_test = X[int(0.8 * len(X)):], y[int(0.8 * len(y)):]  # Assuming 80% training and 20% testing

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}

def get_evaluate_fn(model: DecisionTreeClassifier):
    """Return an evaluation function for server-side evaluation."""

    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, float]
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        task2.set_model_params(model, parameters)  # Update model parameters
        y_pred_proba = model.predict_proba(X_test)  # Get predicted probabilities
        # Calculate log loss; for DecisionTreeClassifier, ensure it's not empty
        if len(y_pred_proba) > 0:
            loss = log_loss(y_test, y_pred_proba)
        else:
            loss = float("inf")  # Handle case with no predictions
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    model = DecisionTreeClassifier()  # Create a DecisionTreeClassifier instance
    task2.set_initial_params(model)  # Set initial parameters (if applicable)

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),  # Use the evaluation function for Decision Tree
        on_fit_config_fn=fit_round,  # Use the fitting round function
    )

    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3)  # Set the number of rounds to 3
    )