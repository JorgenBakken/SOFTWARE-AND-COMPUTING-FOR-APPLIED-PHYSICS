import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

import help_functions as hf
import ship_variables as sv
import create_animation as ca

#---------------------------------------------------------------------------------------------------------------------------

def plot_small_angle_approx(dt):
    '''
    Plot the comparison between the full equation and small angle approximation solutions for a simple harmonic oscillator.

    Inputs:
    dt : Time step for numerical integration

    Returns:
    None (displays the plot)
    '''

    # Create subplots for the plots
    fig, axs = plt.subplots(ncols=2)

    # Define initial angular velocity
    w_0 = np.asarray([20 * np.pi/180, 0])

    # Large angle
    # Solve the ODEs using the full equation and small angle approximation 
    t_matrix, w_matrix = hf.solve_ODE(dt, t_0=0, t_end=20, w_0=w_0, f=hf.two_component_w_RHS, step_function=hf.euler_step)
    t_matrix_small_angle, w_matrix_small_angle = hf.solve_ODE(dt, t_0=0, t_end=20, w_0=w_0,
                                        f=hf.two_component_w_RHS_small_angle, step_function=hf.euler_step)

    # Plot the comparison for greater angle
    axs[0].set_ylabel(r"$\theta$")
    axs[0].set_xlabel(r"$t$")
    axs[0].plot(t_matrix, w_matrix[:,0] * 180 /np.pi, color="b", label=r"$\theta'' \propto \sin \theta$")
    axs[0].scatter(t_matrix_small_angle[::150], w_matrix_small_angle[:,0][::150] * 180 /np.pi, color="r", label=r"$\theta'' \propto \theta$")

    # Small angle
    # Solve the ODEs using the full equation and small angle approximation 
    w_0 = np.asarray([1 * np.pi/180, 0])
    t_matrix, w_matrix = hf.solve_ODE(dt, t_0=0, t_end=20, w_0=w_0, f=hf.two_component_w_RHS, step_function=hf.euler_step)
    t_matrix_small_angle, w_matrix_small_angle = hf.solve_ODE(dt, t_0=0, t_end=20, w_0=w_0,
                                        f=hf.two_component_w_RHS_small_angle, step_function=hf.euler_step)
    # Plot the comparison for smaller angle
    axs[1].set_ylabel(r"$\theta$")
    axs[1].set_xlabel(r"$t$")
    axs[1].plot(t_matrix, w_matrix[:,0] * 180 /np.pi, color="b")  # , label=r"$\theta'' \propto \sin \theta$")
    axs[1].scatter(t_matrix_small_angle[::150], w_matrix_small_angle[:,0][::150] * 180 /np.pi, color="r")  # , label=r"$\theta'' \propto \theta$")
    fig.legend(loc="upper center", ncol=2)

    plt.show()

#---------------------------------------------------------------------------------------------------------------------------

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
    dt_array_actual_euler, error_euler = hf.error_as_func_of_dt(target, dt_array, t_0, t_end, hf.euler_step, theta_0)
    dt_array_actual_RK4, error_RK4 = hf.error_as_func_of_dt(target, dt_array, t_0, t_end, hf.RK4_step, theta_0)

    # Fit a linear function to the logarithm of errors
    popt, pcov = scipy.optimize.curve_fit(hf.linear_function, np.log10(dt_array_actual_euler), np.log10(error_euler))
    popt2, pcov2 = scipy.optimize.curve_fit(hf.linear_function, np.log10(dt_array_actual_RK4), np.log10(error_RK4))

    # Extract the slope from the fit parameters
    a_euler = popt[0]
    a_RK4 = popt2[0]

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(dt_array_actual_euler, error_euler, label="Euler, slope=%.2f" %(a_euler))
    ax.plot(dt_array_actual_RK4, error_RK4, label="RK4, slope=%.2f" %(a_RK4))
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.show()

#---------------------------------------------------------------------------------------------------------------------------

def plot_ship_coordinates(dt):
    '''
    Plot the ship's x-coordinate and y-coordinate as functions of time, taking into account the varying water displacement area.

    Inputs:
    dt      : Time step size

    Returns:
    None
    '''
    # Create subplots for x-coordinate and y-coordinate
    fig, axs = plt.subplots(ncols=2)

    # Set initial conditions
    w_0 = np.asarray([20 * np.pi/180, 0 , sv.yC0, 0, 0, 0])

    # Solve the ODE using RK4 method
    t_array, w_matrix = hf.solve_ODE(dt, t_0=0, t_end=20, w_0=w_0, f=hf.six_component_w_RHS, step_function=hf.RK4_step)

    # Plot x-coordinate
    axs[0].set_ylabel(r"$x$")
    axs[0].set_xlabel(r"$t$")
    axs[0].plot(t_array, w_matrix[:,1], color="b")

    # Plot y-coordinate
    axs[1].set_ylabel(r"$y$")
    axs[1].set_xlabel(r"$t$")
    axs[1].plot(t_array, w_matrix[:,2], color="b")

    # Display the plot
    plt.tight_layout()
    plt.show()

#---------------------------------------------------------------------------------------------------------------------------

def plot_capsizing(dt):
    '''
    Plot the time it takes for the ship to capsize as a function of angular velocity.

    Inputs:
    dt: Time step size

    Returns:
    None
    '''

    # Calculate capsizing 
    capsizing_omega, capsizing_time = hf.calculate_capsizing(dt)

    # Plotting
    plt.plot(capsizing_omega, capsizing_time, 'ro')
    plt.xlabel('Angular Velocity (omega)')
    plt.ylabel('Time to Capsizing (t)')
    plt.title('Time to Capsizing as a Function of Angular Velocity (-1 if it does not capsize)')
    plt.grid(True)
    plt.show()

#---------------------------------------------------------------------------------------------------------------------------

def plot_angle(t, theta1, theta2, label1, label2):
    '''
    Plot the angle of the mass pendulum.

    Inputs:
    - t: An array of time values.
    - theta: An array of angle values.
    - label: A string indicating the label for the plot.

    Returns:
    None
    '''
    plt.plot(t, theta1 * 180 / np.pi, label=label1)
    plt.plot(t, theta2 * 180 / np.pi, label=label2)
    plt.title('Angle')
    plt.xlabel('$t$ (s)', fontsize=14)
    plt.ylabel(r'$\theta$ (degrees)', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()

#---------------------------------------------------------------------------------------------------------------------------

def plot_position(t, s1, s2, label1, label2):
    '''
    Plot the position of the load relative to M.

    Inputs:
    - t: An array of time values.
    - s: An array of position values.
    - label: A string indicating the label for the plot.

    Returns:
    None
    '''
    plt.plot(t, s1, label=label1)
    plt.plot(t, s2, label=label2)
    plt.title('Position of load relative to M')
    plt.xlabel('$t$ (s)', fontsize=14)
    plt.ylabel('$s_L$ (m)', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()

#---------------------------------------------------------------------------------------------------------------------------

def plot_pendulum_results(fence):
    '''
    Plot the results of the mass pendulum simulation.

    Inputs:
    - fence: A boolean indicating whether to include a fence.

    Returns:
    None
    '''

    # Simulate the pendulum for the small and large masses
    t_s, theta_s, x_s, y_s, s_s, omega_s, v_s = hf.calculate_mass_pendulum(mass='small', fence=fence)
    t_l, theta_l, x_l, y_l, s_l, omega_l, v_l = hf.calculate_mass_pendulum(mass='large', fence=fence)

    # Plot angle
    plot_angle(t_l, theta_l, theta_s, label1=r"$m_L = 0.08m$", label2="$m_L = 0.001m$")

    # Plot position of load relative to M
    plot_position(t_l, s_l, s_s, label1=r"$m_L = 0.08m$", label2=r"$m_L = 0.001m$")

#---------------------------------------------------------------------------------------------------------------------------

def plot_damping_effect(theta, omega, k_f_array, dt, t_0, t_end, fence, FW0, y):
    """
    Plot the damping effect on the angle of a system over time.
    
    Inputs:
    theta    : Initial angle
    omega    : Initial angular velocity
    k_f_array: Array of damping coefficients
    dt       : Time step for solving the ODE
    t_0      : Initial time
    t_end    : End time
    fence    : Boolean indicating whether there is a fence
    FW0      : Initial force of the water 
    y        : Initial vertical displacement
    
    Returns:
    None
    """
    w = np.asarray([theta, 0, y, 0, omega, 0, 0, 0])

    for i in range(len(k_f_array)):
        t, w_matrix_RK4 = hf.solve_ODE(dt=dt, t_0=t_0, t_end=t_end, w_0=w,
                                       f=hf.eight_component_w_RHS_extended, step_function=hf.RK4_step,
                                       fence=fence, kf=k_f_array[i], FW0=FW0)

        theta_RK4 = w_matrix_RK4[:, 0]

        plt.plot(t, theta_RK4 * 180 / np.pi, label=r"$k_f = %.1f$" % (k_f_array[i]))

    plt.xlabel('$t$ (s)', fontsize=20)
    plt.ylabel(r'$\theta$ (degrees)', fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.show()

#---------------------------------------------------------------------------------------------------------------------------

def animate_damping_effect(dt, t_0, t_end, step_function, FW0, theta, omega, y, k_f_array):
    """
    Animates the damping effect on the ship's deck movement.

    This function demonstrates the effect of damping on the ship's deck movement by solving the differential equations
    using the Runge-Kutta 4th order method for different damping coefficients.

    Inputs:
    dt             : Time step for solving the ODE
    t_0            : Initial time
    t_end          : End time
    step_function  : ODE integration step function
    FW0            : Fence deflection at t = 0
    theta          : Initial angle
    omega          : Initial angular velocity
    y              : Initial vertical displacement
    k_f_array      : Array of damping coefficients

    Returns:
        None
    """

    w_0_eight = np.asarray([theta, 0, y, 0, omega, 0, 0, 0])  # Initial conditions for the eight-component vector

    # Iterate over the damping coefficients
    for i in range(len(k_f_array)):
        # Solve the ordinary differential equations using the Runge-Kutta 4th order method
        t, w_matrix_RK4 = hf.solve_ODE(dt, t_0, t_end, w_0=w_0_eight,
                                       f=hf.eight_component_w_RHS_extended, step_function=step_function,
                                       fence=False, kf=k_f_array[i], FW0=FW0)

        # Extract the solution components
        theta_RK4 = w_matrix_RK4[:, 0]  # Angular displacement
        x_C_RK4 = w_matrix_RK4[:, 1]  # x-coordinate of the center of mass
        y_C_RK4 = w_matrix_RK4[:, 2]  # y-coordinate of the center of mass
        s_L_RK4 = w_matrix_RK4[:, 3]  # Load position relative to the center of mass
        omega_RK4 = w_matrix_RK4[:, 4]  # Angular velocity
        vL_RK4 = w_matrix_RK4[:, 7]  # Load velocity

        # Animate the deck movement using the extracted solution components
        ca.animate_deck_movement(t, theta_RK4, x_C_RK4, y_C_RK4, s_L_RK4, stepsize=0.01)

#---------------------------------------------------------------------------------------------------------------------------

def animate_angle_with_fence(dt, t_0, t_end, theta, omega, y, kf, step_function, FW0, omegaW):
    """
    Animate and plot the angle of a system with a fence over time.

    Inputs:
    dt: Time step for solving the ODE
    t_0: Initial time
    t_end: End time
    theta: Initial angle
    omega: Initial angular velocity
    y: Initial vertical displacement
    kf: Friction coefficient
    step_function: ODE integration step function
    omegaW : Resonance frequency of the water
    FW0    : Initial force of the water 

    Returns:
    None
    """

    # Set initial conditions
    w_0 = np.asarray([theta, 0, y, 0, omega, 0, 0, 0])

    # Solve the ODE and obtain the time and state matrices
    t, w_matrix_RK4 = hf.solve_ODE(dt, t_0, t_end, w_0, f=hf.eight_component_w_RHS_extended,
                                   step_function=step_function, fence=True, kf=kf, FW0=FW0, omegaW=omegaW)

    # Extract the relevant quantities from the state matrix
    theta_RK4 = w_matrix_RK4[:, 0]
    x_C_RK4 = w_matrix_RK4[:, 1]
    y_C_RK4 = w_matrix_RK4[:, 2]
    s_L_RK4 = w_matrix_RK4[:, 3]

    # Animate the deck movement with the given quantities
    ca.animate_deck_movement(t, theta_RK4, x_C_RK4, y_C_RK4, s_L_RK4, fence=True, stepsize=0.1)


