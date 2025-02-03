import numpy as np
import pandas as pd
from numba import jit


@jit(nopython=True)
def gradient_descent(X, Y, theta, alpha, tolerance):
    m = len(Y)
    iteration = 1
    while True:
        prediction = X @ theta
        error = prediction - Y
        gradient = X.T @ error / m
        new_theta = theta - alpha * gradient
        print(iteration)
        if np.all(np.abs(new_theta - theta) < tolerance):
            break
        iteration += 1
        theta = new_theta
        print(theta)
    print("Finish!")
    
    return theta


def multiple_linear_regression(X_df, Y_df, alpha=0.01, tolerance=1e-6):
    X = X_df.to_numpy()
    Y = Y_df.to_numpy().reshape(-1, 1)
    

    X = np.column_stack((np.ones(X.shape[0]), X))
    theta = np.zeros((X.shape[1], 1))
    #new_values = np.array([0, 0, 0, 0]).reshape(-1, 1) Reconfigurar punto de inicio de ser necesario

    #theta[:new_values.shape[0]] = new_values

    theta = gradient_descent(X, Y, theta, alpha, tolerance)

    return theta