"""sklearnexample: A Flower / scikit-learn app."""

import numpy as np
import pandas as pd  # Import pandas for reading the CSV
from flwr.common import NDArrays
from sklearn.tree import DecisionTreeClassifier

# This information is needed to create a correct scikit-learn model
NUM_UNIQUE_LABELS = 10  # Assuming 10 classes
NUM_FEATURES = 23  # Adjust based on your dataset


def get_model_parameters(model: DecisionTreeClassifier) -> NDArrays:
    """Returns the parameters (tree structure) of a sklearn DecisionTreeClassifier model."""
    return model.tree_.__getstate__()


def set_model_params(model: DecisionTreeClassifier, params: NDArrays) -> DecisionTreeClassifier:
    """Sets the parameters (tree structure) of a sklearn DecisionTreeClassifier model."""
    print(params)
    if isinstance(params, list):
        model.tree_.__setstate__(params[0])  # Adjust based on actual structure
    else:
    # If it's supposed to be a dict, handle that case
        model.tree_.__setstate__(params["tree_state"])  # Assuming it should be this way
    return model


def set_initial_params(model: DecisionTreeClassifier) -> None:
    """Sets initial parameters for the DecisionTreeClassifier model before training."""
    pass  # No specific initialization needed


def create_decision_tree_and_instantiate_parameters(max_depth=None):
    """Helper function to create a DecisionTreeClassifier model."""
    model = DecisionTreeClassifier(
        max_depth=max_depth,  # Set maximum depth as needed
        min_samples_split=2,  # Example hyperparameter
        # random_state=42,  # Optional: for reproducibility
    )
    return model


def load_data(partition_id: int, num_partitions: int):
    """Load data from CSV file and partition it."""
    df = pd.read_csv("new_dataset.csv")  # Load your dataset

    # Split into features and labels
    X = df.iloc[:, :-1].values  # Assuming last column is the label
    y = df.iloc[:, -1].values  # Last column is the label

    # Partition the data manually for federated learning
    # Assuming you have num_partitions clients, you'll slice the data here
    partition_size = len(df) // num_partitions
    start = partition_id * partition_size
    end = (partition_id + 1) * partition_size if partition_id < num_partitions - 1 else len(df)
    
    X_partition = X[start:end]
    y_partition = y[start:end]

    # Split the edge data: 80% train, 20% test
    X_train, X_test = X_partition[: int(0.8 * len(X_partition))], X_partition[int(0.8 * len(X_partition)) :]
    y_train, y_test = y_partition[: int(0.8 * len(y_partition))], y_partition[int(0.8 * len(y_partition)) :]

    return X_train, X_test, y_train, y_test
