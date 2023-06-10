import numpy as np

def gamma_func(theta, beta, radius, dyC):
    '''
    Calculates the sector angle gamma

    theta: rotational angle 
    beta: sector angle
    radius: radius of ship (ship is assumed to a half circle)
    dyC: vertical displacement 
    '''
    arccos_term = np.cos(beta / 2) - 4 * (1 - np.cos(theta)) / (3 * np.pi)
    assert -1 <= arccos_term <= 1, "Value of 'a' is outside the range (-1, 1)"
    
    alpha = 2 * np.arccos(arccos_term)
    assert -1 <= alpha <= 1, "Value of 'alpha' is outside the range (-1, 1)"

    return 2 * np.arccos(np.cos(alpha / 2) + dyC / radius)

def euler_step(f, t, w_old, dt):
    '''
    Solve dw/dt = f(w), where w can be a vector
    Performs one Euler step 

    Inputs 
    f     : function 
    t     : current time
    w_old : initial step 
    dt    : step size 
    w_new : new step
    '''
    w_new = w_old + dt * f(t, w_old)
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


def solve_ODE(dt, t_0, t_end, w_0, f, step_function):
    '''
    dt : step length 
    t_0: start time 
    t_end: end time 
    w_0: start step
    num_step: number of steps
    dt_actual: corrected to be able to seperate each time step in equidistant steps
    w_matrix : a matrix containing all the time steps of w
    '''
    num_steps = find_num_steps(t_end,t_0,dt)
    dt_actual = find_actual_dt(t_end,t_0,dt,num_steps)
    w_matrix = np.zeros(num_steps, len(w_0))
    w_matrix[0,:] = w_0
    for i in range(0, num_steps - 1):
        w_matrix[i + 1] = step_function(f, w_matrix[i], dt_actual)
    return w_matrix