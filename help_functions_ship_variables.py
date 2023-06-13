import numpy as np

def calculate_beta(sigma, sigma0):
    """
    Calculates the value of beta using the iterative method.

    Inputs:
        sigma (float): The value of sigma.
        sigma0 (float): The value of sigma0.

    Returns:
        float: The calculated value of beta.
    """

    max_error = 1E-6  # Maximum acceptable error
    max_iter = int(1E5)  # Maximum number of iterations
    error = 1.0  # Initialize the error
    j = 0  # Initialize the iteration counter
    beta0 = np.pi / 2  # Initial guess for beta

    while error > max_error and j < max_iter:
        beta1 = np.sin(beta0) + np.pi * sigma / sigma0  # Calculate the next value of beta
        error = abs(beta1 - beta0)  # Calculate the error
        beta0 = beta1  # Update the value of beta
        j += 1  # Increment the iteration counter

    return beta0