# Import here whatever you may need
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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

        if np.ndim(X) > 1:
            X = X.reshape(1, -1)
        X_mean=np.mean(X)
        y_mean=np.mean(y)
        self.coefficients = np.sum((X-X_mean)*(y-y_mean))/np.sum((X-X_mean)**2)
        self.intercept = y_mean - self.coefficients*X_mean

    # This part of the model you will only need for the last part of the notebook
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
        
        self.intercept = None
        self.coefficients = None

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

        if np.ndim(X) == 1:
            # Predicción para regresión simple
            predictions = self.intercept + self.coefficients * X
        else:
            # Predicción para regresión múltiple
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
    # Calcular la suma de los residuos al cuadrado
    ss_res = np.sum((y_true - y_pred) ** 2)
    # Calcular la suma total de los cuadrados
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    rmse = np.sqrt(ss_res / len(y_true))
    mae = np.mean(np.abs(y_true - y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}



# ### Scikit-Learn comparison



def sklearn_comparison(x, y, linreg):

    ### Compare your model with sklearn linear regression model
    #  : Import Linear regression from sklearn
    # Asegurarse de que x tenga la forma adecuada para sklearn
    x_reshaped = x.reshape(-1, 1) if x.ndim == 1 else x

   # Crear y entrenar el modelo de scikit-learn
    sklearn_model = LinearRegressor()
    sklearn_model.fit(x_reshaped, y)

    # Now, you can compare coefficients and intercepts between your model and scikit-learn's model
    print("Custom Model Coefficient:", linreg.coefficients)
    print("Custom Model Intercept:", linreg.intercept)
    print("Scikit-Learn Coefficient:", sklearn_model.coef_[0])
    print("Scikit-Learn Intercept:", sklearn_model.intercept_)
    return {
        "custom_coefficient": linreg.coefficients,
        "custom_intercept": linreg.intercept,
        "sklearn_coefficient": sklearn_model.coef_[0],
        "sklearn_intercept": sklearn_model.intercept_,
    }

def anscombe_quartet():
    # Load Anscombe's quartet
    # These four datasets are the same as in slide 19 of chapter 02-03: Linear and logistic regression
    anscombe = sns.load_dataset("anscombe")
    # Obtener los identificadores de cada uno de los 4 conjuntos de datos
    datasets = np.sort(anscombe["dataset"].unique())
    
    models = {}
    results = {"R2": [], "RMSE": [], "MAE": []}
    
    for dataset in datasets:
        # Filtrar el dataset actual
        data = anscombe[anscombe["dataset"] == dataset]
        X = data["x"].values  # Predictor (1D)
        y = data["y"].values  # Respuesta

        # Crear y ajustar el modelo de regresión simple
        model = LinearRegressor()
        model.fit_simple(X, y)
        # Realizar predicciones
        y_pred = model.predict(X)

        # Guardar el modelo para su uso posterior
        models[dataset] = model

        # Imprimir los coeficientes para cada dataset
        print(f"Dataset {dataset}: Coefficient: {model.coefficients}, Intercept: {model.intercept}")

        evaluation_metrics = evaluate_regression(y, y_pred)


        # Print evaluation metrics for each dataset
        print(
            f"R2: {evaluation_metrics['R2']}, RMSE: {evaluation_metrics['RMSE']}, MAE: {evaluation_metrics['MAE']}"
        )
        results["R2"].append(evaluation_metrics["R2"])
        results["RMSE"].append(evaluation_metrics["RMSE"])
        results["MAE"].append(evaluation_metrics["MAE"])
    return results


# Go to the notebook to visualize the results
