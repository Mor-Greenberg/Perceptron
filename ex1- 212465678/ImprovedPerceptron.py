import numpy as np

class ImprovedPerceptron:
    def __init__(self, learning_rate=1.0, max_iter=1000, tolerance=0.01):
        """
        Initialize the improved perceptron with:
        - learning_rate: how much to update weights per mistake
        - max_iter: maximum number of training iterations 
        - tolerance: minimal acceptable error rate before early stopping
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.weights = None  # will hold the learned weight vector

    def fit(self, X, y):
        """
        Train the perceptron on data (X, y) using the Perceptron learning algorithm.
        Stops early if error rate drops below self.tolerance.
        Also keeps the best weights seen so far.
        """
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias column (1s)
        self.weights = np.zeros(X.shape[1])  # Initialize weights to zero

        best_weights = self.weights.copy()
        best_score = 0

        for epoch in range(self.max_iter):
            errors = 0
            for xi, target in zip(X, y):
                prediction = self.predict_row(xi)
                update = self.learning_rate * (target - prediction)
                if update != 0:
                    self.weights += update * xi  # Update weights
                    errors += 1

            error_rate = errors / len(y)

            # Evaluate current model on all data (without bias column)
            current_score = self.score(X[:, 1:], y)
            if current_score > best_score:
                best_score = current_score
                best_weights = self.weights.copy()

            if error_rate <= self.tolerance:
                break  # Early stopping if error is low enough

        self.weights = best_weights  # Use best weights found

    def predict_row(self, x_row):
        """
        Predict label (0 or 1) for a single input vector x_row.
        """
        return 1 if np.dot(x_row, self.weights) >= 0 else 0

    def predict(self, X):
        """
        Predict labels (0 or 1) for all samples in matrix X.
        """
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias
        return np.array([self.predict_row(xi) for xi in X])

    def score(self, X, y):
        """
        Return accuracy: the fraction of correctly classified samples.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
