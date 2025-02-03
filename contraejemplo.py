from GradientDescentOptimized import multiple_linear_regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_data():
    X = np.linspace(-10, 10, 100)  # Range of x values
    Y = ((X + 2)**2 - 3)**2 + X + 2  # Applying the custom function
    return pd.DataFrame(X, columns=['X']), pd.DataFrame(Y, columns=['Y'])

# Generate data
X_df, Y_df = generate_data()

multiple_linear_regression(X_df, Y_df, 0.001)


