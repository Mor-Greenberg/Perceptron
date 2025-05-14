import pandas as pd
import numpy as np
from Perceptron import Perceptron  #Import Perceptron class
from ImprovedPerceptron import ImprovedPerceptron  # Import improvedPerceptron class


# Load the dataset
df = pd.read_csv("Breast Cancer.csv")

# Separate features and labels
X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].values

# Custom split function (60% train, 20% val, 20% test)
def split_dataset(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=None):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_end = int(train_ratio * len(X))
    val_end = train_end + int(val_ratio * len(X))
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]

# Split data
X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y, random_state=42)

# Create and train the model
model = Perceptron()
model.fit(X_train, y_train)

# Evaluate performance
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

# Print results
print("\n--- Regular Perceptron Results (A) ---")

print(f"Train accuracy: {train_acc:.4f}")
print(f"Train error: {1 - train_acc:.4f}")

print(f"Test accuracy: {test_acc:.4f}")
print(f"Test error: {1 - test_acc:.4f}")



# Create and train the improved model
improved_model = ImprovedPerceptron()
improved_model.fit(X_train, y_train)

# Evaluate performance
improved_train_acc = improved_model.score(X_train, y_train)
improved_test_acc = improved_model.score(X_test, y_test)

# Print improved model results
print("\n--- Improved Perceptron Results (B) ---")

print(f"Train accuracy: {improved_train_acc:.4f}")
print(f"Train error: {1 - improved_train_acc:.4f}")

print(f"Test accuracy: {improved_test_acc:.4f}")
print(f"Test error: {1 - improved_test_acc:.4f}")
