import numpy as np
import matplotlib.pyplot as plt

import help_functions as hf
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
plotting.plot_ship_coordinates(dt = 0.01, 
                               t_0=0,
                               t_end=20,
                               theta = 20 * np.pi/180,
                               y = sv.yC0,
                               step_function=hf.RK4_step)

# Plotting the time it takes for the ship to capsize as a function of angular velocity
plotting.plot_capsizing(dt = 0.01)

# Plotting the results of the mass pendulum simulation without a fence
plotting.plot_pendulum_results(fence=False)

# Plotting the results of the mass pendulum simulation with a fence
plotting.plot_pendulum_results(fence=True)

# Plotting the effect of damping on the angle of a system over time
plotting.plot_damping_effect(theta = 0,
                             omega = 0.4,
                             k_f_array = np.asarray([10, 500, 2000, 4000]),
                             dt = 0.01,
                             t_0 = 0,
                             t_end = 180,
                             fence = False,
                             FW0 = 0, 
                             y = sv.yC0)

# Calling the animate_damping_effect function to generate an animation illustrating the effect of damping on the system over time. 
# This function simulates the motion of a mass pendulum system with different damping coefficients and produces an animation to visually demonstrate the behavior. 
# The animation showcases the movement of the pendulum and how it responds to the presence of damping.
plotting.animate_damping_effect(dt=0.01, 
                                t_0=0, 
                                t_end=180,
                                step_function=hf.RK4_step,
                                FW0=0,
                                theta = 0,
                                omega = 0.4, 
                                y = sv.yC0,
                                k_f_array = np.asarray([10, 500, 2000, 4000]))

# Calling the animate_angle_with_fence function to generate an animation and plot the angle of a system with a fence over time.
# This function animates the movement of the system's deck and simultaneously plots the angle variation.
plotting.animate_angle_with_fence(dt = 0.01,
                                  t_0 = 0,
                                  t_end = 240,
                                  theta = 0,
                                  omega = 2.0 * np.pi / 180,
                                  y = sv.yC0,
                                  kf = 100,
                                  step_function=hf.RK4_step, 
                                  FW0=sv.FW0,
                                  omegaW=sv.omegaW)

