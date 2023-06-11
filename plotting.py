import numpy as np
import matplotlib.pyplot as plt

import help_functions as hf

def plot_small_angle_approx(dt, m, h, IC):
    # Create subplots for the plots
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
    
    # Define initial angular velocity and time range
    w_0 = np.asarray([20 * np.pi / 180, 0])
    t_0, t_end = 0, 20
    
    # Solve the ODEs using the full equation and small angle approximation
    t_matrix, w_matrix = hf.solve_ODE(dt, t_0, t_end, w_0, m, h, IC, f=hf.two_component_w_RHS, step_function=hf.euler_step)
    t_matrix_small_angle, w_matrix_small_angle = hf.solve_ODE(dt, t_0, t_end, w_0, m, h, IC, f=hf.two_component_w_RHS_small_angle, step_function=hf.euler_step)

    # Plot the comparison for greater angle
    axs[0].set_title("Comparison for Greater Angle")
    axs[0].set_ylabel(r"$\theta$")
    axs[0].set_xlabel(r"$t$")
    axs[0].plot(t_matrix, w_matrix[:, 0] * 180 / np.pi, color="blue", linewidth=0.5, label=r"$\theta'' \propto \sin \theta$")
    axs[0].scatter(t_matrix_small_angle[::150], w_matrix_small_angle[:, 0][::150] * 180 / np.pi, color="red", marker=".", s=10)

    # Define initial angular velocity for smaller angle
    w_0 = np.asarray([1 * np.pi / 180, 0])

    # Solve the ODEs using the full equation and small angle approximation for smaller angle
    t_matrix, w_matrix = hf.solve_ODE(dt, t_0, t_end, w_0, m, h, IC, f=hf.two_component_w_RHS, step_function=hf.euler_step)
    t_matrix_small_angle, w_matrix_small_angle = hf.solve_ODE(dt, t_0, t_end, w_0, m, h, IC, f=hf.two_component_w_RHS_small_angle, step_function=hf.euler_step)

    # Plot the comparison for smaller angle
    axs[1].set_title("Comparison for Smaller Angle")
    axs[1].set_ylabel(r"$\theta$")
    axs[1].set_xlabel(r"$t$")
    axs[1].plot(t_matrix, w_matrix[:, 0] * 180 / np.pi, color="blue", linewidth=0.5)
    axs[1].scatter(t_matrix_small_angle[::150], w_matrix_small_angle[:, 0][::150] * 180 / np.pi, color="red", marker=".", s=10)

    fig.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    plt.show()


# Define the constants
R = 10  # Ship radius (m)
Askip = 0.5 * np.pi * R ** 2  # Ship cross-sectional area (m^2)
sigma = 500  # Ship mass density (kg/m^2)
m = sigma * Askip  # Ship mass
dt = 0.001  # Time step size
h = 4 * R / (3 * np.pi)  # Distance M - C
IC = 0.5 * m * R ** 2 * (1 - 32 / (9 * np.pi ** 2)) #The ship's moment of inertia with respect to the axis through C

plot_small_angle_approx(dt, m, h, IC)