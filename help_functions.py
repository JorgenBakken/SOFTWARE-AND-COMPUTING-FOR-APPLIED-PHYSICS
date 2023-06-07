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