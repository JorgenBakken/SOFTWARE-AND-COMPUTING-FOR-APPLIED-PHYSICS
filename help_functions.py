import numpy as np
import scipy.constants

def gamma_func(theta, beta, radius, dyC):
    '''
    Calculates the sector angle gamma

    theta: rotational angle 
    beta: sector angle
    radius: radius of ship (ship is assumed to a half circle)
    dyC: vertical displacement 
    '''
    arccos_term_1 = np.cos(beta / 2) - 4 * (1 - np.cos(theta)) / (3 * np.pi)
    assert -1 <= arccos_term_1 <= 1, "Value of 'arccosterm_1' is outside the range [-1, 1]"
    
    alpha = 2 * np.arccos(arccos_term_1)
    arccos_term_2 = np.cos(alpha / 2) + dyC / radius
    assert -1 <= arccos_term_2 <= 1, "Value of 'arccosterm_2' is outside the range [-1, 1]"

    return 2 * np.arccos(arccos_term_2)

def euler_step(f, t, w_old, m, h, IC, dt):
    '''
    Solve dw/dt = f(w), where w can be a vector
    Performs one Euler step 

    Inputs:
    f      : Function that represents the derivative dw/dt
    t      : Current time
    w_old  : Initial step
    m      : Mass of the ship
    h      : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC     : The ship's moment of inertia with respect to the axis through C
    dt     : Step size

    Returns:
    w_new  : New step after performing the Euler step
    '''
    w_new = w_old + dt * f(t, w_old, m, h, IC)
    return w_new


def find_num_steps( t_0 ,t_end, dt):
    '''
    Function to calculate how many steps we are able to take
    We make sure that we are able to perform full step
    t_0 : start time 
    t_end : end time
    dt : step size 
    '''
    return int((t_end - t_0)/dt) + 1

def find_actual_dt(t_0, t_end, num_steps):
    '''
    Function that corrects the step size such that we can move from t_0 to t_end by only full steps 
    t_0 : start time 
    t_end : end time
    num_steps : number of full steps performed
    '''
    return (t_end - t_0) / num_steps


def solve_ODE(dt, t_0, t_end, w_0, m, h, IC, f, step_function):
    '''
    dt : step length 
    t_0: start time 
    t_end: end time 
    w_0: start step
    num_step: number of steps
    t_array : vector containing all the time step
    m      : Mass of the ship
    h      : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC     : The ship's moment of inertia with respect to the axis through C
    dt_actual: corrected to be able to seperate each time step in equidistant steps
    w_matrix : a matrix containing all the time steps of w
    '''
    num_steps = find_num_steps(t_0,t_end,dt)

    t_array, dt_actual = np.linspace(t_0, t_end, num_steps + 1, retstep=True)

    w_matrix = np.zeros((num_steps + 1, len(w_0)))
    w_matrix[0,:] = w_0

    for i in range(0, num_steps):
        w_matrix[i + 1] = step_function(f, t_array[i], w_matrix[i], m, h, IC, dt_actual)
    return t_array, w_matrix


# Implement f 

def two_component_w_RHS(t, w, m, h, IC):
    '''
    Function that calculates the derivative of w (dw/dt) for a two-component system.
    The function represents the right-hand side (RHS) of the differential equation.

    Inputs:
    t    : Current time
    w    : Vector [w_0, w_1]
           w_0 represents the angle theta
           w_1 represents the angular velocity omega
    m    : Mass of the ship
    h    : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC   : The ship's moment of inertia with respect to the axis through C

    Returns:
    updated_w : Array containing the updated values of [w_0, w_1]
    '''

    # Gravitational constant
    gravity_constant = scipy.constants.g

    # Initialize array for updated values
    updated_w = np.zeros(2)

    # Calculate the derivative of theta (dw_0/dt)
    updated_w[0] = w[1]

    # Calculate the derivative of omega (dw_1/dt)
    updated_w[1] = -m * gravity_constant * h * np.sin(w[0]) / IC

    return updated_w

def two_component_w_RHS_small_angle(t, w, m, h, IC):
    '''
    Function that calculates the derivative of w (dw/dt) for a two-component system using the small angle approximation.
    The function represents the right-hand side (RHS) of the differential equation.

    Inputs:
    t    : Current time
    w    : Vector [w_0, w_1]
           w_0 represents the angle theta
           w_1 represents the angular velocity omega
    m    : Mass of the ship
    h    : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC   : The ship's moment of inertia with respect to the axis through C

    Returns:
    updated_w : Array containing the updated values of [w_0, w_1]
    '''

    # Gravitational constant
    gravity_constant = scipy.constants.g

    # Initialize array for updated values
    updated_w = np.zeros(2)

    # Calculate the derivative of theta (dw_0/dt)
    updated_w[0] = w[1]

    # Apply small angle approximation for the derivative of omega (dw_1/dt)
    updated_w[1] = -m * gravity_constant * h * w[0] / IC

    return updated_w



