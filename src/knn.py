import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

class KNNClassifier:

    """
    A simple K-Nearest Neighbors (KNN) classifier implementation.
    
    Parameters:
        k (int): The number of neighbors to consider.
        distance (str): The distance metric to use ("euc" for Euclidean or "man" for Manhattan).
        test_size (float): The proportion of the data to be used for testing in fit_eval() method.
        seed (int): The random seed for train-test splitting.
    """

    def __init__(self, k:int = 3, distance:str = "euc", test_size:float = None, seed=0) -> None:

        """
        Initialize the KNNClassifier with specified parameters.
        """

        self._init_error_check(k, distance, test_size, seed)
        
        self.k = k
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.fit_eval_called = False
        self.distance = 1 if distance == "man" else 2
        self.test_size = test_size
        self.seed = seed
        

    def _init_error_check(self, k, distance, test_size, seed) -> None:

        """
        Check the validity of the initialization parameters.
        
        Parameters:
            k (int): The number of neighbors to consider.
            distance (str): The distance metric to use ("euc" for Euclidean or "man" for Manhattan).
            test_size (float): The proportion of the data to be used for testing.
            seed (int): The random seed for train-test splitting.
        
        Raises:
            ValueError: If any of the parameter values are invalid.
        """

        if k < 1:
            raise ValueError(
                "k must be a positive integer"
            )

        if not distance in ["euc", "man"]:
            raise ValueError(
                f"distance parameter can only be 'euc' or 'man', your input: {distance}"
            )
        
        if test_size:
            if not isinstance(test_size, float):
                raise ValueError(
                    "test_size must be a floating point number between 0 and 1"
                )
            
            if test_size < 0 or test_size > 1:
                raise ValueError(
                    "test_size should greater than 0 and less than 1"
                )
            
        if not isinstance(seed, int):
            raise ValueError(
                "seed should be an integer"
            )
        
        if seed < 0:
            raise ValueError(
                "seed should be a positive integer"
            )
            
    def fit(self, X, y) -> None:

        """
        Fit the KNN model with training data.
        
        Parameters:
            X (array-like): Training data features.
            y (array-like): Training data labels.
        """

        self._check_fit_error(X, y)
        
        if self.test_size:
            raise RuntimeError(
                "Use fit_eval() method if you have specified test_size parameter"
            )

        else:
            self.X_train = X
            self.y_train = y

    def get_train_Xy(self) -> tuple:

        """
        Get the training data features and labels.
        
        Returns:
            tuple: A tuple containing training data features and labels.
        """

        self._raise_get_error()
        return self.X_train, self.y_train
    
    def get_test_Xy(self) -> tuple:

        """
        Get the testing data features and labels.
        
        Returns:
            tuple: A tuple containing testing data features and labels.
        """

        self._raise_get_error()
        return self.X_test, self.y_test
    
    def _raise_get_error(self) -> None:

        """
        Raise errors for get_train_Xy and get_test_Xy methods when used incorrectly.
        
        Raises:
            ValueError: If called without appropriate conditions.
            RuntimeError: If called without fitting and evaluating the model first.
        """

        if not self.test_size:
            raise ValueError(
                "You have not opted to split the data using test_size parameter, \
                    hence this method is not applicable unless you specify a test_size for splitting your data"
            )

        if not self.fit_eval_called:
            raise RuntimeError(
                "Use fit_eval() before getting test or train data"
            )

    def fit_eval(self, X, y) -> None:

        """
        Fit and evaluate the KNN model with training and testing data.
        
        Parameters:
            X (array-like): Features of the entire dataset.
            y (array-like): Labels of the entire dataset.
        """

        self._check_fit_error(X, y)

        if not self.test_size:
            raise ValueError(
                "Specify a test_size before using fit_eval() method"
            )

        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.seed
            )

        print(self.evaluate(self.X_test, self.y_test))

        self.fit_eval_called = True

    def _check_fit_error(self, X, y) -> None:

        """
        Check for errors in the data and labels provided for fitting the model.
        
        Parameters:
            X (array-like): Training data features.
            y (array-like): Training data labels.
        
        Raises:
            ValueError: If there are inconsistencies or issues in the data and labels.
        """

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if isinstance(y, pd.Series):
            y.to_numpy()

        if len(y.shape) != 1:
            raise ValueError(
                f"y can only be a single column, you have passed a dataframe with dimensions {y.shape}"
            )
        
        if not int(np.unique(np.isnan(X)) == [False]):
            raise ValueError(
                "NaN values found in the dataset, please impute NaN values before training"
            )

        if len(X) != len(y):
            raise ValueError(
                f"Length of X_train is {len(X)} and Length of X_train is {len(y)}, lengths don't match"
            )
        
        if len(X) < self.k:
            raise ValueError(
                f"Sample size is lesser than the number of neighbors mentioned: \
                      samples: {len(X)}, Neighbors: {self.k}"
            )

    def predict(self, X) -> list:

        """
        Predict labels for input data points.
        
        Parameters:
            X (array-like): Data points to be predicted.
            
        Returns:
            list: Predicted labels for each data point.
        """

        if self.X_train is None or self.y_train is None:
            raise RuntimeError(
                "Fit the model using '.fit()' method before using predict"
            )
        
        predictions = [self._predict(x) for x in X]

        return predictions

    def _predict(self, x) -> int:
        
        """
        Make a prediction for a single data point.
        
        Parameters:
            x (array-like): The data point to predict.
            
        Returns:
            int: The predicted label for the given data point.
        """
        
        distances = [self._minkowski_distance(x, x_train, p=self.distance) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_lab = [self.y_train[i] for i in k_indices]

        majority_label = Counter(k_lab).most_common()[0][0]

        return majority_label

    def _minkowski_distance(self, x1, x2, p=2) -> float:

        """
        Calculate the Minkowski distance between two data points.
        
        Parameters:
            x1 (array-like): First data point.
            x2 (array-like): Second data point.
            p (int, optional): The power parameter for the Minkowski distance calculation.
            
        Returns:
            float: The Minkowski distance between x1 and x2.
        """

        distance = np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1/p)
        return distance

    
    def evaluate(self, X_test, y_test) -> tuple | float:

        """
        Evaluate the model's performance on test data.
        
        Parameters:
            X_test (array-like): Features of the test data.
            y_test (array-like): Labels of the test data.
            
        Returns:
            pandas.DataFrame or float: If binary classification, returns a DataFrame 
            containing accuracy, precision, recall, and F1-score. If not, returns accuracy.
        """

        if len(X_test) != len(y_test):
            raise ValueError(
                f"Length of X_test and y_test don't match: len(X_test) = {len(X_test)} \
                    and len(y_test) = {len(y_test)}"
            )
        
        y_hat = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_hat)

        if len(np.unique(y_test)) == 2:
            precision = precision_score(y_test, y_hat)
            f1 = f1_score(y_test, y_hat)
            recall = recall_score(y_test, y_hat)

            score_df = pd.DataFrame(
                {
                    "Metric": [
                        "Accuracy", 
                        "Precision", 
                        "Recall", 
                        "F1"
                    ],
                    "Score": [
                        accuracy,
                        precision,
                        f1,
                        recall
                    ]
                }
            ).set_index("Metric", drop=True)

            return score_df, self._roc_plot(y_test, y_hat)

        return accuracy

    def _roc_plot(self, y_test, y_hat):

        """
        Plot the Receiver Operating Characteristic (ROC) curve.
        
        Parameters:
            y_test (array-like): True labels of the test data.
            y_hat (array-like): Predicted labels of the test data.
        """

        fpr, tpr, _ = roc_curve(y_test, y_hat)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.grid(True)
        plt.show()