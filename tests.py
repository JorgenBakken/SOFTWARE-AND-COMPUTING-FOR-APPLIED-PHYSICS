import help_functions as hf

import numpy as np
import scipy.constants

from hypothesis import given,settings
from hypothesis import strategies as st

@given(theta=st.floats(min_value=-np.pi*0.5, max_value=np.pi*0.5),
       beta=st.floats(min_value= np.pi*0.5, max_value=np.pi),
       radius=st.floats(min_value=0.0000000001, max_value=10000),
       dyC_factor=st.floats(min_value=-0.25, max_value=0.25))
def test_gamma_func(theta, beta, radius, dyC_factor):
    '''
    Test the gamma_func function, to make sure it returns a float value

    Arguments:
    - theta: Rotational angle of boat.
    - beta: Sector angle of where boat touches the water in equilibrium.
    - radius: Radius of the ship, assumed to be positive.
    - dyC_factor: Factor for calculating dyC based on the radius.

    Test:
    - Checks if the function returns a float.
    - Assumes:
        - radius > 0.
        - gamma > 0.
        - -pi/2 < theta < pi/2.
        - pi/2 < beta < pi.
        - -radius/4 < dyC < radius/4
    '''
    dyC = radius * dyC_factor
    gamma = hf.gamma_func(theta, beta, radius, dyC)
    assert isinstance(gamma, float), "Gamma should be a float value"

@given(t_0 = st.floats(min_value=-10000000,max_value=10000000), 
       t_end_extra = st.floats(min_value=0.000001, max_value=10000000), 
       dt = st.floats(min_value= 0.000001, max_value=1000000))
def test_find_num_steps(t_0 ,t_end_extra, dt):
    '''
    Tests the find_num_steps to make sure it returns a valid number of steps
    It should either be the number of steps required to get to the end time, or past it

    Arguments 
    t_0         : start time 
    t_end_extra : distance from t_0 to t_end, to make sure t_end comes after t_0
    t_end       : end time
    dt          : step size 
    '''
    t_end = t_0 + t_end_extra
    num_steps = hf.find_num_steps(t_0 ,t_end, dt)
    print(":")
    print(num_steps)
    print(dt)
    print(t_end_extra)
    assert isinstance(num_steps, int), "Number of steps should be an integer"
    assert t_0 + num_steps * dt >= t_end, "Should be able to reach the end"

@given(t_0 = st.floats(min_value=-10000000,max_value=10000000), 
       t_end_extra = st.floats(min_value=0.000001, max_value=10000000), 
       num_steps = st.integers(min_value= 1, max_value=1000000))
def test_find_actual_dt(t_0, t_end_extra, num_steps):
    '''
    Test the correctness of the find_actual_dt function.
    It checks whether the computed time step correctly reaches the specified end time
    by comparing the expected end time with the actual end time.
    
    Parameters:
    t_0 (float)        : Start time
    t_0 (float)        : End time 
    t_end_extra (float): Extra time added to the start time to compute the end time
    num_steps (int)    : Number of steps
    '''
    t_end = t_0 + t_end_extra
    tolerance = 0.0000001
    new_dt = hf.find_actual_dt(t_0, t_end, num_steps)
    assert isinstance(new_dt, float), "The new dt should be a flaot"
    assert abs(t_0 + new_dt * num_steps - t_end) < tolerance, "Should reach the end point approximately"

@given(t  = st.floats(min_value=-10000000, max_value=1000000),
       w_old  = st.floats(min_value=-10000000, max_value=1000000),
       dt = st.floats(min_value=0.000001, max_value=1000000),       
       m=st.floats(min_value=0.0001, max_value=1000000),
       h=st.floats(min_value=0.0, max_value=1000),
       IC=st.floats(min_value=0.000001, max_value=100.0))
def test_euler_step(t, w_old, m, h, IC, dt):
    '''
    Test the euler_step function
    Creates a function f, finds the value after one step, compare the euler step to this value

    Inputs:
    t         : Current time
    w_old     : Initial step
    m         : Mass of the ship
    h         : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC        : The ship's moment of inertia with respect to the axis through C
    dt        : Step size
    f         : Test function representing the derivative dw/dt
    w_correct : Expected result computed using the Euler method
    w_euler   : Result obtained from the euler_step function
    '''
    def f(t, w_old, m, h, IC):
        return np.sin(t) + w_old

    w_correct = w_old + dt * f(t, w_old, m, h, IC)

    w_euler = hf.euler_step(f, t, w_old, m, h, IC, dt)

    assert np.isclose(w_correct, w_euler), "The euler step should give a value close to the calculated step"

@given(dt    = st.floats(min_value=0.000001, max_value=10000000),
       t_0   = st.floats(min_value=-10000000, max_value=10000000),
       t_end_factor = st.floats(min_value= 0.1, max_value=10),
       w_0   = st.lists(st.floats(min_value=-10000000, max_value=10000000),min_size=2, max_size=2),
       m=st.floats(min_value=0.0001, max_value=1000000),
       h=st.floats(min_value=0.0, max_value=1000),
       IC=st.floats(min_value=0.000001, max_value=100.0))
@settings(max_examples=50)
def test_solve_ODE_shape_t_array(dt, t_0, t_end_factor, w_0, m, h, IC):
    '''
    Test the solve_ODE function
    Check the shape of the time array

    Inputs:
    dt             : Step length
    t_0            : Start time
    t_end          : End time
    t_end_factor   : factor to multiply with dt to get t_end, ensures not to many steps 
    w_0            : Initial step
    m         : Mass of the ship
    h         : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC        : The ship's moment of inertia with respect to the axis through C

    Asserts:
    - The shape of the returned time array matches the expected shape
    '''
    t_end = t_0 + t_end_factor * dt

    def f(t, w_old, m, h, IC):
        return np.sin(t) + w_old
    
    t_array, w_matrix = hf.solve_ODE(dt, t_0, t_end, w_0, m, h, IC, f, hf.euler_step)
    num_steps = hf.find_num_steps(t_0 ,t_end, dt)
    assert t_array.shape[0] == num_steps + 1

@given(dt    = st.floats(min_value=0.000001, max_value=10000000),
       t_0   = st.floats(min_value=-10000000, max_value=10000000),
       t_end_factor = st.floats(min_value= 0.000001, max_value=10),
       w_0   = st.lists(st.floats(min_value=-10000000, max_value=10000000)),
       m=st.floats(min_value=0.0001, max_value=1000000),
       h=st.floats(min_value=0.0, max_value=1000),
       IC=st.floats(min_value=0.000001, max_value=100.0))
@settings(max_examples=50)
def test_solve_ODE_shape_sol_array(dt, t_0, t_end_factor, w_0, m, h, IC):
    '''
    Test the solve_ODE function
    Check the shape of the solution matrix 

    Inputs:
    dt             : Step length
    t_0            : Start time
    t_end          : End time
    t_end_factor   : factor to multiply with dt to get t_end, ensures not to many steps 
    w_0            : Initial step
    m              : Mass of the ship
    h              : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC             : The ship's moment of inertia with respect to the axis through C

    Asserts:
    - The shape of the returned solution array matches the expected shape
    '''
    t_end = t_0 + t_end_factor * dt

    def f(t, w_old, m, h, IC):
        return np.sin(t) + w_old

    t_array, w_matrix = hf.solve_ODE(dt, t_0, t_end, w_0, m, h, IC, f, hf.euler_step)
    assert w_matrix.shape == (len(t_array), len(w_0))

@given(dt    = st.floats(min_value=0.000001, max_value=10000000),
       t_0   = st.floats(min_value=-10000000, max_value=10000000),
       t_end_factor = st.floats(min_value= 0.000001, max_value=10),
       w_0   = st.lists(st.floats(min_value=-10000000, max_value=10000000)),
       m=st.floats(min_value=0.0001, max_value=1000000),
       h=st.floats(min_value=0.0, max_value=1000),
       IC=st.floats(min_value=0.000001, max_value=100.0))
@settings(max_examples=50)
def test_solve_ODE_t0(dt, t_0, t_end_factor, w_0, m, h, IC):
    '''
    Test the solve_ODE function
    Check the first element of the time array

    Inputs:
    dt             : Step length
    t_0            : Start time
    t_end          : End time
    t_end_factor   : factor to multiply with dt to get t_end, ensures not to many steps 
    w_0            : Initial step
    m              : Mass of the ship
    h              : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC             : The ship's moment of inertia with respect to the axis through C

    Asserts:
    - The first element of the time array matches the start time
    '''
    t_end = t_0 + t_end_factor * dt
    def f(t, w_old, m, h, IC):
        return np.sin(t) + w_old
     
    t_array, w_matrix = hf.solve_ODE(dt, t_0, t_end, w_0, m, h, IC, f, hf.euler_step)
    assert np.isclose(t_array[0], t_0)


@given(dt    = st.floats(min_value=0.000001, max_value=10000000),
       t_0   = st.floats(min_value=-10000000, max_value=10000000),
       t_end_factor = st.floats(min_value= 0.000001, max_value=10),
       w_0   = st.lists(st.floats(min_value=-10000000, max_value=10000000),min_size = 1),
       m=st.floats(min_value=0.0001, max_value=1000000),
       h=st.floats(min_value=0.0, max_value=1000),
       IC=st.floats(min_value=0.000001, max_value=100.0))
@settings(max_examples=50)
def test_solve_ODE_t_end(dt, t_0, t_end_factor, w_0, m, h, IC):
    '''
    Test the solve_ODE function
    Check the last element of the time array

    Inputs:
    dt             : Step length
    t_0            : Start time
    t_end          : End time
    t_end_factor   : factor to multiply with dt to get t_end, ensures not to many steps 
    w_0            : Initial step
    m              : Mass of the ship
    h              : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC             : The ship's moment of inertia with respect to the axis through C

    Asserts:
    - The last element of the time array matches the end time
    '''
    t_end = t_0 + t_end_factor * dt
    def f(t, w_old, m, h, IC):
        return np.sin(t) + w_old
     
    t_array, w_matrix = hf.solve_ODE(dt, t_0, t_end, w_0, m, h, IC, f, hf.euler_step)
    assert np.isclose(t_array[-1], t_end)


@given(dt    = st.floats(min_value=0.000001, max_value=10000000),
       t_0   = st.floats(min_value=-10000000, max_value=10000000),
       t_end_factor = st.floats(min_value= 0.000001, max_value=10),
       w_0   = st.lists(st.floats(min_value=-10000000, max_value=10000000),min_size = 1),
       m=st.floats(min_value=0.0001, max_value=1000000),
       h=st.floats(min_value=0.0, max_value=1000),
       IC=st.floats(min_value=0.000001, max_value=100.0))
@settings(max_examples=50)
def test_solve_ODE_stepvalues(dt, t_0, t_end_factor, w_0, m, h, IC):
    '''
    Test the solve_ODE function
    Check the solution matrix for each step

    Inputs:
    dt             : Step length
    t_0            : Start time
    t_end          : End time
    t_end_factor   : factor to multiply with dt to get t_end, ensures not to many steps 
    w_0            : Initial step
    m              : Mass of the ship
    h              : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC             : The ship's moment of inertia with respect to the axis through C

    Asserts:
    - The solution matrix has the correct values for each step
    '''
    t_end = t_0 + t_end_factor * dt
    def f(t, w_old, m, h, IC):
        return np.sin(t) + w_old
     
    t_array, w_matrix = hf.solve_ODE(dt, t_0, t_end, w_0, m, h, IC, f, hf.euler_step)

    num_steps = hf.find_num_steps(t_0, t_end, dt)
    new_dt = hf.find_actual_dt(t_0, t_end, num_steps)

    assert np.allclose(w_matrix[0], w_0), "w_matrix[0] == w_0"

    for i in range(len(t_array) - 1):
        assert np.allclose(w_matrix[i + 1], hf.euler_step(f, t_array[i], w_matrix[i], m, h, IC, new_dt))


# Test the two_component_w_RHS

@given(t=st.floats(min_value=0.0, max_value=100.0),
       w=st.lists(st.floats(min_value=-np.pi, max_value=np.pi), min_size=2, max_size=2),
       m=st.floats(min_value=0.0001, max_value=1000000),
       h=st.floats(min_value=0.0, max_value=1000),
       IC=st.floats(min_value=0.00001, max_value=100.0))
def test_two_component_w_RHS_length(t, w, m, h, IC):
    '''
    Test for the two_component_w_RHS function.

    Inputs:
    t    : Current time
    w    : Vector [w_0, w_1] representing initial values of theta and omega
    m    : Mass of the ship
    h    : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC   : The ship's moment of inertia with respect to the axis through C

    Checks:
    - Checks if the output has a length of 2.
    '''

    # Calculate the derivative using the function
    updated_w = hf.two_component_w_RHS(t, w, m, h, IC)

    assert len(updated_w) == 2,  "the updated w should have length 2"


@given(t=st.floats(min_value=0.0, max_value=100.0),
       w=st.lists(st.floats(min_value=-np.pi, max_value=np.pi), min_size=2, max_size=2),
       m=st.floats(min_value=0.0001, max_value=1000000),
       h=st.floats(min_value=0.0, max_value=1000),
       IC=st.floats(min_value=0.000001, max_value=100.0))
def test_two_component_w_RHS_theta_update(t, w, m, h, IC):
    '''
    Test for the two_component_w_RHS function.

    Inputs:
    t    : Current time
    w    : Vector [w_0, w_1] representing initial values of theta and omega
    m    : Mass of the ship
    h    : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC   : The ship's moment of inertia with respect to the axis through C

    Checks:
    - Checks if the calculated derivative of theta is equal to the given value of omega.
    '''

    # Calculate the derivative using the function
    updated_w = hf.two_component_w_RHS(t, w, m, h, IC)

    assert np.isclose(updated_w[0], w[1])

    # Calculate the expected derivative of omega
    gravity_constant = scipy.constants.g
    expected_w1 = -m * gravity_constant * h * np.sin(w[0]) / IC

    # Check if the calculated derivative of omega matches the expected value
    assert np.isclose(updated_w[1], expected_w1)

@given(t=st.floats(min_value=0.0, max_value=100.0),
       w=st.lists(st.floats(min_value=-np.pi, max_value=np.pi), min_size=2, max_size=2),
       m=st.floats(min_value=0.0001, max_value=1000000),
       h=st.floats(min_value=0.0, max_value=1000),
       IC=st.floats(min_value=0.000001, max_value=100.0))
def test_two_component_w_RHS_omega_update(t, w, m, h, IC):
    '''
    Test for the two_component_w_RHS function.

    Inputs:
    t    : Current time
    w    : Vector [w_0, w_1] representing initial values of theta and omega
    m    : Mass of the ship
    h    : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC   : The ship's moment of inertia with respect to the axis through C

    Checks:
    - Checks if the calculated derivative of omega follows the expected formula.
    '''

    # Calculate the derivative using the function
    updated_w = hf.two_component_w_RHS(t, w, m, h, IC)

    # Calculate the expected derivative of omega
    gravity_constant = scipy.constants.g
    expected_w1 = -m * gravity_constant * h * np.sin(w[0]) / IC

    # Check if the calculated derivative of omega matches the expected value
    assert np.isclose(updated_w[1], expected_w1)


# Test the two_component_w_RHS_small_angle

@given(t=st.floats(min_value=0.0, max_value=100.0),
       w=st.lists(st.floats(min_value=-np.pi, max_value=np.pi), min_size=2, max_size=2),
       m=st.floats(min_value=0.0001, max_value=1000000),
       h=st.floats(min_value=0.0, max_value=1000),
       IC=st.floats(min_value=0.000001, max_value=100.0))
def test_two_component_w_RHS_small_angle_length(t, w, m, h, IC):
    '''
    Test for the two_component_w_RHS_small_angle function.

    Inputs:
    t    : Current time
    w    : Vector [w_0, w_1] representing initial values of theta and omega
    m    : Mass of the ship
    h    : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC   : The ship's moment of inertia with respect to the axis through C

    Checks:
    - The output has a length of 2
    '''

    # Calculate the derivative using the function
    updated_w = hf.two_component_w_RHS_small_angle(t, w, m, h, IC)

    assert len(updated_w) == 2,  "the updated w should have length 2"

@given(t=st.floats(min_value=0.0, max_value=100.0),
       w=st.lists(st.floats(min_value=-np.pi, max_value=np.pi), min_size=2, max_size=2),
       m=st.floats(min_value=0.0001, max_value=1000000),
       h=st.floats(min_value=0.0, max_value=1000),
       IC=st.floats(min_value=0.000001, max_value=100.0))
def test_two_component_w_RHS_small_angle_omega_update(t, w, m, h, IC):
    '''
    Test for the two_component_w_RHS_small_angle function.

    Inputs:
    t    : Current time
    w    : Vector [w_0, w_1] representing initial values of theta and omega
    m    : Mass of the ship
    h    : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC   : The ship's moment of inertia with respect to the axis through C

    Checks:
    - The calculated derivative of omega follows the expected formula based on the small angle approximation
    '''

    # Calculate the derivative using the function
    updated_w = hf.two_component_w_RHS_small_angle(t, w, m, h, IC)

    # Calculate the expected derivative of omega based on the small angle approximation
    expected_w1 = -m * scipy.constants.g * h * w[0] / IC

    # Check if the calculated derivative of omega matches the expected value
    assert np.isclose(updated_w[1], expected_w1)


@given(t=st.floats(min_value=0.0, max_value=100.0),
       w=st.lists(st.floats(min_value=-np.pi, max_value=np.pi), min_size=2, max_size=2),
       m=st.floats(min_value=0.0001, max_value=1000000),
       h=st.floats(min_value=0.0, max_value=1000),
       IC=st.floats(min_value=0.000001, max_value=100.0))
def test_two_component_w_RHS_small_angle_theta_update(t, w, m, h, IC):
    '''
    Test for the two_component_w_RHS_small_angle function.

    Inputs:
    t    : Current time
    w    : Vector [w_0, w_1] representing initial values of theta and omega
    m    : Mass of the ship
    h    : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC   : The ship's moment of inertia with respect to the axis through C

    Checks:
    - The calculated derivative of theta is equal to the given value of omega
    '''

    # Calculate the derivative using the function
    updated_w = hf.two_component_w_RHS_small_angle(t, w, m, h, IC)

    # Check if the calculated derivative of theta matches the expected value
    assert np.isclose(updated_w[0], w[1]), "updated_w[0] == w[1]"