import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

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
A_ship = 0.5 * np.pi * R ** 2  # Ship cross-sectional area (m^2)
sigma = 500  # Ship mass density (kg/m^2)
m = sigma * A_ship  # Ship mass
dt = 0.001  # Time step size
h = 4 * R / (3 * np.pi)  # Distance M - C
IC = 0.5 * m * R ** 2 * (1 - 32 / (9 * np.pi ** 2)) #The ship's moment of inertia with respect to the axis through C

#plot_small_angle_approx(dt, m, h, IC)


def plot_error_step_function(dt_array, theta_0, omega_freq):
    '''
    Plot the error as a function of the time step size for Euler and RK4 methods.

    Inputs:
    dt_array    : Array of time step sizes.
    theta_0     : Initial angle value.
    omega_freq  : Angular frequency.

    Returns: 
    Nothing 
    '''
    t_0 = 0
    t_end = 20
    target = hf.analytical_solution_small_angle(t_end, theta_0, omega_freq)
    
    # Calculate errors for Euler and RK4 methods
    dt_array_actual_euler, error_euler = hf.error_as_func_of_dt(target, dt_array, t_0, t_end, hf.euler_step, theta_0, m, h, IC)
    dt_array_actual_RK4, error_RK4 = hf.error_as_func_of_dt(target, dt_array, t_0, t_end, hf.RK4_step, theta_0, m, h, IC)
    
    # Fit a linear function to the logarithm of errors
    popt, pcov = scipy.optimize.curve_fit(hf.linear_function, np.log10(dt_array_actual_euler), np.log10(error_euler))
    popt2, pcov2 = scipy.optimize.curve_fit(hf.linear_function, np.log10(dt_array_actual_RK4), np.log10(error_RK4))
    
    # Extract the slope from the fit parameters
    slope_euler = popt[0]
    slope_RK4 = popt2[0]

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(dt_array_actual_euler, error_euler, label="Euler step, slope=%.2f" % slope_euler)
    ax.plot(dt_array_actual_RK4, error_RK4, label="RK4 step, slope=%.2f" % slope_RK4)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Step Size (dt)")
    ax.set_ylabel("Error")
    ax.set_title("Error for different step functions as a result of step size")
    ax.legend()
    plt.show()


gravity_constant = scipy.constants.g
dt_array = np.logspace(-1.5, -3.5, 20)
theta_0 = 0.01
omega_freq = np.sqrt(m*gravity_constant*h/IC)
plot_error_step_function(dt_array, theta_0, omega_freq)
