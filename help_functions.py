import numpy as np
import ship_variables as sv

def gamma_func(theta, dyC):
    """
    Calculates the value of gamma given the angle theta and the vertical displacement dyC.

    Inputs:
    theta  : Angle theta
    dyC    : Vertical displacement dyC

    Returns:
    gamma  : Value of gamma
    """
    arccos_term_1 = np.cos(sv.beta / 2) - 4 * (1 - np.cos(theta)) / (3 * np.pi)
    assert np.logical_and(arccos_term_1 >= -1, arccos_term_1 <= 1).any(), "Value of 'arccosterm_1' is outside the range [-1, 1]"
    alpha = 2 * np.arccos(arccos_term_1)

    arccos_term_2 = np.cos(alpha / 2) + dyC / sv.R
    assert np.logical_and(arccos_term_2 >= -1, arccos_term_2 <= 1).any(), "Value of 'arccosterm_2' is outside the range [-1, 1]"

    gamma = 2 * np.arccos(arccos_term_2)

    return gamma

def euler_step(f, t, w, dt, fence=False, kf=sv.kf, omegaW=sv.omegaW, FW0=sv.FW0):
    """
    Performs one Euler step to solve the ODE dw/dt = f(w).

    Inputs:
    f      : Function that represents the derivative dw/dt
    t      : Current time
    w      : Initial step
    dt     : Step size
    fence  : Boolean flag indicating whether the fence is present (default: False)
    kf     : Friction constant for the fence (default: sv.kf)
    omegaW : Resonance frequency of the water (default: sv.omegaW)
    FW0    : Initial force of the water (default: sv.FW0)

    Returns:
    w_new  : New step after performing the Euler step
    """

    w_new = w + dt * f(t, w, fence, kf=kf, omegaW=omegaW, FW0=FW0)
    return w_new


def RK4_step(f, t, w, dt, fence=False, kf=sv.kf, omegaW=sv.omegaW, FW0=sv.FW0):
    """
    Performs one step using the fourth-order Runge-Kutta (RK4) method to solve the ODE dw/dt = f(w).

    Inputs:
    f      : Function that represents the derivative dw/dt
    t      : Current time
    w      : Initial step
    dt     : Step size
    fence  : Boolean flag indicating whether the fence is present (default: False)
    kf     : Friction constant for the fence (default: sv.kf)
    omegaW : Resonance frequency of the water (default: sv.omegaW)
    FW0    : Initial force of the water (default: sv.FW0)

    Returns:
    w_new  : New step after performing the RK4 step
    """

    k1 = f(t, w, fence, kf=kf, omegaW=omegaW, FW0=FW0)
    k2 = f(t + dt/2.0, w + (dt / 2) * k1, fence, kf=kf, omegaW=omegaW, FW0=FW0)
    k3 = f(t + dt/2.0, w + (dt / 2) * k2, fence, kf=kf, omegaW=omegaW, FW0=FW0)
    k4 = f(t + dt, w + dt * k3, fence, kf=kf, omegaW=omegaW, FW0=FW0)
    w_new = w + (dt / 6) * (k1 + (2 * k2) + (2 * k3) + k4)
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


def solve_ODE(dt, t_0, t_end, w_0, f, step_function=RK4_step, fence=False, kf=sv.kf, omegaW=sv.omegaW, FW0=sv.FW0):
    """
    Solves an ODE where the right-hand side is described by the function f using a specified step function.

    Inputs:
    dt             : Step size
    t_0            : Initial time
    t_end          : End time
    w_0            : Initial step
    f              : Function that represents the derivative dw/dt
    step_function  : Step function to use for solving the ODE (default: RK4_step)
    fence          : Boolean flag indicating whether the fence is present (default: False)
    kf             : Friction constant for the fence (default: sv.kf)
    omegaW         : Resonance frequency of the water (default: sv.omegaW)
    FW0            : Initial force of the water (default: sv.FW0)

    Returns:
    t_array        : Array of time values
    w_matrix       : Matrix of step values corresponding to each time value
    """

    load_off = False
    number_steps = int((t_end - t_0) / dt) + 1  # Find the number of discrete t-values
    # It's likely that dt doesn't divide evenly into t_0, so we find the dt value that is closest
    t_array, dt_actual = np.linspace(t_0, t_end, number_steps, retstep=True)
    # This is important, initializing w in this way makes the function work, regardless of the number of components in w
    w_matrix = np.zeros((len(t_array), len(w_0)))
    number_of_components = len(w_0)  # If greater than 6, we have a load

    w_matrix[0] = w_0

    for i in range(0, len(t_array) - 1):
        w_matrix[i + 1] = step_function(f, t_array[i], w_matrix[i], dt_actual, fence, kf=kf, omegaW=omegaW, FW0=FW0)

        # Capsizing functionality
        if number_of_components > 2:
            dyC = w_matrix[i + 1, 2] - sv.yC0
            gamma = gamma_func(w_matrix[i + 1, 0], dyC)
            if np.abs(w_matrix[i + 1, 0]) > (np.pi - gamma) / 2:  
                w_matrix[i + 1:, 0] = np.sign(w_matrix[i, 0]) * np.pi / 2 * np.ones(len(w_matrix[i + 1:, 0]))
                break

        if number_of_components > 6:
            if np.abs(w_matrix[i + 1, 3]) > sv.R and not load_off:
                if fence:
                    w_matrix[i + 1, 3] = np.sign(w_matrix[i + 1, 3]) * sv.R
                    w_matrix[i + 1, 7] = 0  # We neglect the force transfer from the load to the fence
                else:
                    # If the load falls off the deck, the mass is set to 0 inside the f-function
                    # But we mark where this happens so that we can set the coordinates of the load to a constant value after it has fallen off
                    load_off = True
                    t_load_off = i + 1

    if load_off:
        # The choice of coordinates for the fallen off load is somewhat arbitrary, I chose 3*R
        # where the sign indicates which side the load fell off on
        w_matrix[t_load_off:, 3] = np.sign(w_matrix[t_load_off, 3]) * 3 * sv.R * np.ones(len(w_matrix[t_load_off:, 3]))
        w_matrix[t_load_off:, 7] = np.zeros(len(w_matrix[t_load_off:, 7]))

    return t_array, w_matrix


def two_component_w_RHS(t, w, fence=False, kf=sv.kf, omegaW=sv.omegaW, FW0=sv.FW0):
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
    # Initialize array for updated values
    updated_w = np.zeros(2) 
    
    # Calculate the derivative of theta (dw_0/dt)
    updated_w[0] = w[1]

    # Calculate the derivative of omega (dw_1/dt)
    updated_w[1] = -sv.m * sv.g * sv.h * np.sin(w[0]) / sv.IC

    return updated_w

def two_component_w_RHS_small_angle(t, w, fence=False, kf=sv.kf, omegaW=sv.omegaW, FW0=sv.FW0):
    '''
    Function that calculates the derivative of w (dw/dt) for a two-component system using the small angle approximation.
    The function represents the right-hand side (RHS) of the differential equation.

    Inputs:
    t    : Current time
    w    : Vector [w_0, w_1]
           w_0 represents the angle theta
           w_1 represents the angular velocity omega
    fence: Boolean indicating whether the fence is present (default is False)
    kf   : Spring constant of the fence (default is sv.kf)
    omegaW: Angular velocity of the wind (default is sv.omegaW)
    FW0  : Initial wind force (default is sv.FW0)

    Returns:
    updated_w : Array containing the updated values of [w_0, w_1]
    '''

    # Initialize array for updated values
    updated_w = np.zeros(2) 

    # Calculate the derivative of theta (dw_0/dt)
    updated_w[0] = w[1]

    # Apply small angle approximation for the derivative of omega (dw_1/dt)
    updated_w[1] = - sv.m * sv.g * sv.h * w[0] / sv.IC

    return updated_w

def error_as_func_of_dt(target, dt_array, t_0, t_end, step_function, theta_0):
    '''
    Calculate the error between the computed solution and a target value for different step sizes (dt).

    Inputs:
    target          : The target value or true solution to compare against.
    dt_array       : An array of different step sizes (dt) to test.
    t_0            : The initial time of the system.
    t_end          : The end time of the system.
    step_function  : The numerical integration method to use (e.g., Euler's method, RK4).
    theta_0        : The initial value of the angle (theta).

    Returns:
    actual_dt_array : Array containing the actual step sizes used by the numerical method.
    error_array     : Array containing the absolute errors between the computed solution and the target value.
    '''

    # Initialize arrays for storing actual step sizes and errors
    error_array = np.zeros(len(dt_array))
    actual_dt_array = np.zeros(len(dt_array))

    # Convert theta_0 to a numpy array for compatibility with solve_ODE function
    w_0 = np.asarray([theta_0, 0])

    # Iterate over the dt values
    for i in range(len(dt_array)):
        # Solve the ODE using the specified step size and step function
        t, w_matrix = solve_ODE(dt_array[i], t_0, t_end, w_0, two_component_w_RHS_small_angle,
                                step_function=step_function, fence=False)
        
        # Calculate the absolute error between the computed solution and the target value
        error_array[i] = np.abs(w_matrix[-1, 0] - target)
        
        # Store the actual step size used by the numerical method
        actual_dt_array[i] = t[1] - t[0]

    return actual_dt_array, error_array

def linear_function(x, a, b):
    '''
    Compute the value of a linear function given the input x.

    Inputs:
    x : Input value.
    a : Coefficient of the linear term.
    b : Constant term.

    Returns:
    Result of the linear function evaluation.
    '''
    return a * x + b

def analytical_solution_small_angle(t, theta_0, omega_freq):
    '''
    Calculate the analytical solution for the angle of a simple harmonic oscillator with small angle approximation.

    Inputs:
    t          : Time array
    theta_0    : Initial angle
    omega_freq : Angular frequency

    Returns:
    The analytical solution
    '''
    return theta_0 * np.cos(t * omega_freq)

def six_component_w_RHS(t, w, fence=False, kf=sv.kf, omegaW=sv.omegaW, FW0=sv.FW0):
    '''
    Calculate the derivatives of the state variables for a system of six differential equations.

    Inputs:
    t       : Current time
    w       : Vector representing the state variables [theta, x, y, s, omega, v_x, v_y, v_s]
    fence   : Boolean indicating if there is a fence
    kf      : Stiffness coefficient
    omegaW  : Excitation frequency
    FW0     : Excitation force

    Returns:
    Array containing the updated derivatives of the state variables
    '''

    updated_w = np.zeros(6)

    updated_w[0] = w[3]   # Derivative of theta (angular velocity)
    updated_w[1] = w[4]   # Derivative of x-coordinate
    updated_w[2] = w[5]   # Derivative of y-coordinate

    # Auxiliary variables
    dyC = w[2] - sv.yC0                                     # Vertical displacement of the center of mass
    gamma = gamma_func(w[0], dyC)                           # Angle gamma
    A_water = 0.5 * sv.R ** 2 * (gamma - np.sin(gamma))     # Area of the submerged part of the ship
    FB = sv.sigma0 * A_water * sv.g                         # Buoyant force
    Fy = FB - sv.m * sv.g                                   # Vertical force

    updated_w[3] = -FB * sv.h * np.sin(w[0]) / sv.IC   # Derivative of angular velocity (omega dot)
    updated_w[4] = 0                                   # x-velocity (v_x) remains constant (assuming no external forces)
    updated_w[5] = Fy / sv.m                           # Derivative of y-velocity (v_y)

    return updated_w


def eight_component_w_RHS(t, w, m, h, IC, yC0, sigma0, R, mL, fence=False):
    '''
    Calculate the derivatives of the state variables for a system of eight differential equations.

    Inputs:
    t       : Current time
    w       : Vector representing the state variables [theta, x, y, s, omega, v_x, v_y, v_s]
    m       : Mass of the ship
    h       : Distance between the midpoint of the deck and the ship's center of mass
    IC      : Ship's moment of inertia with respect to the axis through the center of mass
    yC0     : Ship's center of gravity (y-coordinate)
    sigma0  : Water mass density (kg/m^2 per meter length)
    R       : Radius of the ship
    mL      : Mass of the load
    fence   : Boolean indicating if there is a fence

    Returns:
    Array containing the updated derivatives of the state variables
    '''

    # Gravitational constant
    gravitation_constant = scipy.constants.g

    # Initialize the array for updated derivatives
    updated_w = np.zeros(8)

    # Derivative of theta (angular velocity)
    updated_w[0] = w[4]

    # Derivative of x-coordinate
    updated_w[1] = w[5]

    # Derivative of y-coordinate
    updated_w[2] = w[6]

    # Derivative of s
    updated_w[3] = w[7]

    # Auxiliary variables

    # Vertical distance between the ship's center of gravity and the water surface
    dyC = w[2] - yC0

    # Angle of the circular segment of water displaced by the ship
    gamma = gamma_func(w[0], dyC, R, dyC)

    # Area of the circular segment of water displaced by the ship
    A_water = 0.5 * R ** 2 * (gamma - np.sin(gamma))

    # Buoyant force on the ship
    F_B = sigma0 * A_water * gravitation_constant

    # Torque due to buoyant force
    tauB = -F_B * h * np.sin(w[0])

    # Vertical force on the ship
    F_y = F_B - m * gravitation_constant

    # Force from the external load
    N = -mL * gravitation_constant * np.cos(w[0])

    # Torque from the external load
    tauL = N * w[3]

    # Total torque on the ship (including torque from buoyant force and external load)
    tauTOT = tauB + tauL

    # Horizontal force on the ship
    F_x = N * np.sin(w[0])

    # Derivative of angular velocity (omega dot)
    updated_w[4] = tauTOT / IC

    # Derivative of x-velocity (v_x)
    updated_w[5] = F_x / m

    # Derivative of y-velocity (v_y)
    updated_w[6] = F_y / m

    # Derivative of s-velocity (v_s)
    updated_w[7] = -gravitation_constant * np.sin(w[0])

    # Check if load falls off
    if np.abs(w[3]) > R and not fence and mL > 0:
        mL = 0.0

    return updated_w
