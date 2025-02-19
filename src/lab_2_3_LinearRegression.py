# Import here whatever you may need
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class LinearRegressor:
    """
    Linear Regression model that can perform both simple and multiple linear regression.

    Attributes:
        coefficients (np.ndarray): Coefficients of the independent variables in the regression model.
        intercept (float): Intercept of the regression model.
    """
    def __init__(self):
        """Initializes the LinearRegressor model with default coefficient and intercept values."""
        self.coefficients = None
        self.intercept = None

    def fit_simple(self, X, y):
        """
        Fit the model using simple linear regression (one independent variable).

        This method calculates the coefficients for a linear relationship between
        a single predictor variable X and a response variable y.

        Args:
            X (np.ndarray): Independent variable data (1D array).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Asegurarse de que X sea un array 1D
        if np.ndim(X) > 1:
            X = X.reshape(1, -1)

        # Calcular medias
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        # Calcular el coeficiente (pendiente)
        # Formula: sum( (X - X_mean)*(y - y_mean) ) / sum( (X - X_mean)^2 )
        self.coefficients = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean) ** 2)

        # Calcular el intercepto
        self.intercept = y_mean - self.coefficients * X_mean

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array where each column is a variable).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Insertar una columna de 1's para representar el intercepto
        X_with_ones = np.insert(X, 0, 1, axis=1)

        # Ecuación normal: theta = (X^T X)^(-1) X^T y
        theta = np.linalg.inv(X_with_ones.T @ X_with_ones) @ (X_with_ones.T @ y)

        # El primer valor de theta es el intercepto
        self.intercept = theta[0]
        # El resto son los coeficientes
        self.coefficients = theta[1:]


    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        # Distinción entre regresión simple (X 1D) y múltiple (X 2D)
        if np.ndim(X) == 1:
            # Regresión simple
            predictions = self.intercept + self.coefficients * X
        else:
            # Regresión múltiple
            predictions = self.intercept + X.dot(self.coefficients)
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """
    # Suma de los residuos al cuadrado
    ss_res = np.sum((y_true - y_pred) ** 2)
    # Suma total de cuadrados
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Evitar división por cero si ss_tot es 0
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    rmse = np.sqrt(ss_res / len(y_true))
    mae = np.mean(np.abs(y_true - y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}



# ### Scikit-Learn comparison

def sklearn_comparison(x, y, linreg):
    """
    Compara un modelo personalizado (linreg) con el modelo de scikit-learn (LinearRegression).
    """

    # Asegurar que x sea 2D para scikit-learn
    x_reshaped = x.reshape(-1, 1) if x.ndim == 1 else x

    # Crear y entrenar el modelo de scikit-learn
    sklearn_model = LinearRegression()
    sklearn_model.fit(x_reshaped, y)

    # Imprimir y recopilar resultados
    print("Custom Model Coefficients:", linreg.coefficients)
    print("Custom Model Intercept:", linreg.intercept)
    print("Scikit-Learn Coefficients:", sklearn_model.coef_)
    print("Scikit-Learn Intercept:", sklearn_model.intercept_)

    # En caso de regresión simple (1 dimensión), sklearn_model.coef_ es un array de un solo elemento
    # Si el test espera un escalar, usamos sklearn_model.coef_[0]
    if sklearn_model.coef_.size == 1:
        sklearn_coef = sklearn_model.coef_[0]
    else:
        sklearn_coef = sklearn_model.coef_

    return {
        "custom_coefficient": linreg.coefficients,
        "custom_intercept": linreg.intercept,
        "sklearn_coefficient": sklearn_coef,
        "sklearn_intercept": sklearn_model.intercept_,
    }


def anscombe_quartet():
    anscombe = sns.load_dataset("anscombe")
    datasets = np.sort(anscombe["dataset"].unique())
    
    models = {}
    results = {"R2": [], "RMSE": [], "MAE": []}
    
    for dataset in datasets:
        data = anscombe[anscombe["dataset"] == dataset]
        X = data["x"].values
        y = data["y"].values

        model = LinearRegressor()
        model.fit_simple(X, y)
        y_pred = model.predict(X)

        models[dataset] = model
        evaluation_metrics = evaluate_regression(y, y_pred)

        results["R2"].append(evaluation_metrics["R2"])
        results["RMSE"].append(evaluation_metrics["RMSE"])
        results["MAE"].append(evaluation_metrics["MAE"])

    # Ahora devolvemos los cuatro objetos que el test espera
    return anscombe, datasets, models, results

# Go to the notebook to visualize the results
