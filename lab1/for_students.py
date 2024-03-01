import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()    # actual output
x_test = test_data['Weight'].to_numpy()


def extend_with_zeros(array: np.array) -> np.ndarray:
    # Reshape the vector to be a column vector
    zeros_column = array.reshape(-1, 1)

    # Stack the vector_column and ones_column horizontally
    return np.hstack((np.ones_like(zeros_column), zeros_column))


def calculate_mse(t_matrix, x_matrix, y_matrix):
    return sum([((t_matrix[0] + t_matrix[1] * x) - y) ** 2 for x, y in zip(x_matrix, y_matrix)]) / len(x_matrix)


x_train_extended = extend_with_zeros(x_train)

# Compute X^T X
XtX = np.dot(x_train_extended.T, x_train_extended)

# Compute the inverse of X^T X
XtX_inv = np.linalg.inv(XtX)

# Compute (X^T X)^{-1} X^T
XtX_inv_Xt = np.dot(XtX_inv, x_train_extended.T)

# Compute theta
theta_best = np.dot(XtX_inv_Xt, y_train)    # predicted output

print(f"calculated MSE: {calculate_mse(theta_best, x_test, y_test)}")

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

x_test_normalized = (x_test - np.average(x_test)) / np.std(x_test)
y_test_normalized = (y_test - np.average(y_test)) / np.std(y_test)

x_train_normalized = (x_train - np.average(x_train)) / np.std(x_train)
y_train_normalized = (y_train - np.average(y_train)) / np.std(y_train)

x_train_extended_normalized = extend_with_zeros(x_train_normalized)

# Define your initial guess for theta
theta = [0, 0]
for i in range(100):

    f = 2 / len(x_train_extended_normalized)
    c1 = np.matmul(x_train_extended_normalized, theta)
    c2 = np.subtract(c1, y_train_normalized)
    c3 = np.matmul(x_train_extended_normalized.T, c2)
    gradient = f * c3

    theta = theta - 0.1 * gradient

theta_best = theta

print(f"gradient MSE: {calculate_mse(theta_best, x_test_normalized, y_test_normalized)}")

# plot the regression line
x = np.linspace(min(x_test_normalized), max(x_test_normalized), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test_normalized, y_test_normalized)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
