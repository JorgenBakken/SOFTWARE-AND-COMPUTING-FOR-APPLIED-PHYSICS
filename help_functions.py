import numpy as np

#Will need a sector angle gamma for calculations
def gamma_func(theta, beta, radius, dyC):
    '''
    theta: rotational angle 
    beta: sector angle
    radius: radius of ship (ship is assumed to a half circle)
    dyC: vertical displacement 
    '''
    alpha = 2 * np.arccos(np.cos(beta / 2) - 4 * (1 - np.cos(theta)) / (3 * np.pi))
    return 2 * np.arccos(np.cos(alpha / 2) + dyC / radius)


#Need an ode solver 
def euler_step(f, w_old, dt):
    '''
    Solve dw/dt = f(w), where w can be a vector
    f     : function 
    w_old : initial step 
    dt    : step size 
    w_new : new step
    '''
    w_new = w_old + dt * f(w_old)
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