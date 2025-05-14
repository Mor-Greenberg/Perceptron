import numpy as np

 
class Perceptron:

    def __init__(self, learning_rate=1.0, max_iter=1000):
        """
        Initialize the perceptron with a learning rate and maximum number of iterations.
        This also defines the weights field in the class.

        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        
    
    def fit(self, X, y):
        """
        Train the perceptron using the input data X and labels y.
        """
        # Add bias term to each input sample 
        X = np.c_[np.ones(X.shape[0]), X]

        # Initialize weight vector with zeros (including bias)
        self.weights = np.zeros(X.shape[1])

        # Iterate over the data multiple times
        for _ in range(self.max_iter):
            errors = 0
            for xi, target in zip(X, y):
                # Calculate prediction error
                update = self.learning_rate * (target - self.predict_row(xi))
                if update != 0:
                    self.weights += update * xi  # Update weights
                    errors += 1
            if errors == 0:
                break  # Stop early if no errors
    

    def predict_row(self, x_row):
            """
            Predict label (0 or 1) for a single input vector x_row.
            """
            return 1 if np.dot(x_row, self.weights) >= 0 else 0

    def predict(self, X):
            """
            Predict labels for a matrix X of input samples.
            """
            X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
            return np.array([self.predict_row(xi) for xi in X])

    def score(self, X, y):
        """
        Return accuracy score (fraction of correctly classified samples).
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


    