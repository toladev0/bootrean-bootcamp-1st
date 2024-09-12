import numpy as np
from sklearn.linear_model import LinearRegression

# Design matrix (X) with a column of ones for the intercept
X = np.array([
    [1, 2000, 3],
    [1, 2500, 4],
    [1, 2700, 4],
    [1, 2200, 3],
    [1, 3000, 5]
])

# Output vector (y)
y = np.array([250, 300, 320, 280, 350])

# Calculate the coefficients using the matrix formula
coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

# Print the matrix calculation coefficients
print("Matrix Calculation - Intercept (b0):", coefficients[0])
print("Matrix Calculation - Coefficient for Square Footage (b1):", coefficients[1])
print("Matrix Calculation - Coefficient for Bedrooms (b2):", coefficients[2])

# For sklearn, we need to remove the intercept column from X
X_sklearn = X[:, 1:]  # Removing the column of ones

# Create the model and fit it
model = LinearRegression().fit(X_sklearn, y)

# Get the coefficients
sklearn_intercept = model.intercept_
sklearn_coefficients = model.coef_

# Print the sklearn coefficients
print("Sklearn - Intercept (b0):", sklearn_intercept)
print("Sklearn - Coefficient for Square Footage (b1):", sklearn_coefficients[0])
print("Sklearn - Coefficient for Bedrooms (b2):", sklearn_coefficients[1])
