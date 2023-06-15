import numpy as np
import help_functions as hf
import matplotlib.pyplot as plt

import ship_variables as sv
import ship_variables as gv
import plotting

# Plotting the comparison between the full equation and small angle approximation for a simple harmonic oscillator
plotting.plot_small_angle_approx(dt=0.001)

# Defining the parameters for the error plot
dt_array = np.logspace(-1.5, -3.5, 20)
theta_0 = 0.01
omega_freq = np.sqrt(sv.m * sv.g * sv.h / sv.IC)

# Plotting the error as a function of the time step size for Euler and RK4 methods
plotting.plot_error_step_function(dt_array, theta_0, omega_freq)

# Plotting the ship's x-coordinate and y-coordinate as functions of time
plotting.plot_ship_coordinates(dt=0.01)

# Plotting the time it takes for the ship to capsize as a function of angular velocity
plotting.plot_capsizing(dt=0.01)

# Plotting the results of the mass pendulum simulation without a fence
plotting.plot_pendulum_results(fence=False)

# Plotting the results of the mass pendulum simulation with a fence
plotting.plot_pendulum_results(fence=True)