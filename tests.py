import help_functions as hf
import ship_variables as sv

import numpy as np
import scipy.constants

from hypothesis import given,settings
from hypothesis import strategies as st

@given(theta=st.floats(min_value=-np.pi*0.5, max_value=np.pi*0.5),
       dyC=st.floats(min_value=-sv.R * 0.25, max_value=sv.R * 0.25))
def test_gamma_func(theta, dyC):
    '''
    Test the gamma_func function, to make sure it returns a float value

    Arguments:
    - theta: Rotational angle of the boat.
    - dyC: Vertical displacement of the center of gravity of the boat.

    Test:
    - Checks if the function returns a float.
    - Assumes:
        - -pi/2 < theta < pi/2 (within the valid range of theta).
        - -radius/4 < dyC < radius/4 (within the valid range of dyC).
    '''
    gamma = hf.gamma_func(theta, dyC)
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


@given(t=st.floats(min_value=-10000000, max_value=1000000),
       w_old=st.floats(min_value=-10000000, max_value=1000000),
       dt=st.floats(min_value=0.000001, max_value=1000000),
       fence=st.booleans(),
       kf=st.floats(min_value=-10000000, max_value=1000000),
       omegaW=st.floats(min_value=-10000000, max_value=1000000),
       FW0=st.floats(min_value=-10000000, max_value=1000000))
def test_euler_step(t, w_old, dt, fence, kf, omegaW, FW0):
    '''
    Test the euler_step function

    Inputs:
    t         : Current time
    w_old     : Initial step
    dt        : Step size
    fence     : Boolean flag indicating whether the fence is present
    kf        : Friction constant for the fence
    omegaW    : Resonance frequency of the water
    FW0       : Initial force of the water
    '''
    def f(t, w, fence, kf, omegaW, FW0):
        # Test function
        return np.sin(t) + w

    w_correct = w_old + dt * f(t, w_old, fence, kf, omegaW, FW0)

    w_euler = hf.euler_step(f, t, w_old, dt, fence, kf, omegaW, FW0)

    assert np.isclose(w_correct, w_euler), "The euler step should give a value close to the calculated step"



@given(t=st.floats(min_value=-10000000, max_value=1000000),
       w_old=st.floats(min_value=-10000000, max_value=1000000),
       dt=st.floats(min_value=0.000001, max_value=1000000),
       fence=st.booleans(),
       kf=st.floats(min_value=-10000000, max_value=1000000),
       omegaW=st.floats(min_value=-10000000, max_value=1000000),
       FW0=st.floats(min_value=-10000000, max_value=1000000))
def test_RK4_step(t, w_old, dt, fence, kf, omegaW, FW0):
    '''
    Test the RK4 step function

    Inputs:
    t         : Current time
    w_old     : Initial step
    dt        : Step size
    fence     : Boolean flag indicating whether the fence is present
    kf        : Friction constant for the fence
    omegaW    : Resonance frequency of the water
    FW0       : Initial force of the water
    '''
    def f(t, w, fence, kf, omegaW, FW0):
        # Test function 
        return np.sin(t) + w

    k1 = dt * f(t, w_old, fence, kf, omegaW, FW0)
    k2 = dt * f(t + 0.5 * dt, w_old + 0.5 * k1, fence, kf, omegaW, FW0)
    k3 = dt * f(t + 0.5 * dt, w_old + 0.5 * k2, fence, kf, omegaW, FW0)
    k4 = dt * f(t + dt, w_old + k3, fence, kf, omegaW, FW0)

    w_correct = w_old + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

    w_rk4 = hf.RK4_step(f, t, w_old, dt, fence, kf, omegaW, FW0)

    assert np.isclose(w_correct, w_rk4), "The RK4 step should give a value close to the calculated step"


@given(dt=st.floats(min_value=0.000001, max_value=10000000),
       t_0=st.floats(min_value=-10000000, max_value=10000000),
       t_end_factor=st.floats(min_value=0.1, max_value=10),
       w_0=st.lists(st.floats(min_value=-10000000, max_value=10000000), min_size=2, max_size=2),
       fence=st.booleans())
@settings(max_examples=50)
def test_solve_ODE_shape_t_array(dt, t_0, t_end_factor, w_0, fence):
    '''
    Test the solve_ODE function
    Check the shape of the time array

    Inputs:
    dt             : Step length
    t_0            : Start time
    t_end_factor   : Factor to multiply with dt to get t_end, ensures not too many steps
    w_0            : Initial step
    m              : Mass of the ship
    h              : Distance between the midpoint of the deck, M, and the ship's center of mass, C (M - C)
    IC             : The ship's moment of inertia with respect to the axis through C

    Asserts:
    - The shape of the returned time array matches the expected shape
    '''
    t_end = t_0 + t_end_factor * dt

    def f(t, w, fence, kf, omegaW, FW0):
        # Test function 
        return np.sin(t) + w

    t_array, w_matrix = hf.solve_ODE(dt, t_0, t_end, w_0, f, hf.RK4_step, fence)
    num_steps = int(np.ceil((t_end - t_0) / dt)) + 1
    assert len(t_array) == num_steps


@given(dt=st.floats(min_value=0.000001, max_value=10000000),
       t_0=st.floats(min_value=-10000000, max_value=10000000),
       t_end_factor=st.floats(min_value=0.1, max_value=10),
       w_0=st.lists(st.floats(min_value=-10000000, max_value=10000000), min_size=2, max_size=2),
       fence = st.booleans())
@settings(max_examples=50)
def test_solve_ODE_shape_sol_array(dt, t_0, t_end_factor, w_0, fence):
    '''
    Test the solve_ODE function
    Check the shape of the solution matrix 

    Inputs:
    dt             : Step length
    t_0            : Start time
    t_end          : End time
    t_end_factor   : Factor to multiply with dt to get t_end, ensures not too many steps 
    w_0            : Initial step
    fence          : Boolean value if the is a fence

    Asserts:
    - The shape of the returned solution array matches the expected shape
    '''
    t_end = t_0 + t_end_factor * dt

    def f(t, w, fence, kf, omegaW, FW0):
        # Test function 
        return np.sin(t) + w

    t_array, w_matrix = hf.solve_ODE(dt, t_0, t_end, w_0, f, hf.RK4_step, fence, sv.kf, sv.omegaW, sv.FW0)
    assert w_matrix.shape == (len(t_array), len(w_0))

@given(dt=st.floats(min_value=0.000001, max_value=10000000),
       t_0=st.floats(min_value=-10000000, max_value=10000000),
       t_end_factor=st.floats(min_value=0.1, max_value=10),
       w_0=st.lists(st.floats(min_value=-10000000, max_value=10000000), min_size=2, max_size=2),
       fence = st.booleans())
@settings(max_examples=50)
def test_solve_ODE_t0(dt, t_0, t_end_factor, w_0, fence):
    '''
    Test the solve_ODE function
    Check the first element of the time array

    Inputs:
    dt             : Step length
    t_0            : Start time
    t_end          : End time
    t_end_factor   : Factor to multiply with dt to get t_end, ensures not too many steps 
    w_0            : Initial step
    fence          : Boolean value if the is a fence

    Asserts:
    - The first element of the time array matches the start time
    '''
    t_end = t_0 + t_end_factor * dt

    def f(t, w, fence, kf, omegaW, FW0):
        # Test function 
        return np.sin(t) + w
     
    t_array, w_matrix = hf.solve_ODE(dt, t_0, t_end, w_0, f, hf.RK4_step, fence, sv.kf, sv.omegaW, sv.FW0)
    assert np.isclose(t_array[0], t_0)


@given(dt=st.floats(min_value=0.1, max_value=10000000),
       t_0=st.floats(min_value=-10000000, max_value=10000000),
       t_end_factor=st.floats(min_value=1, max_value=10),
       w_0=st.lists(st.floats(min_value=-10000000, max_value=10000000), min_size=2, max_size=2),
       fence = st.booleans())
@settings(max_examples=50)
def test_solve_ODE_t_end(dt, t_0, t_end_factor, w_0, fence):
    '''
    Test the solve_ODE function
    Check the last element of the time array

    Inputs:
    dt             : Step length
    t_0            : Start time
    t_end          : End time
    t_end_factor   : Factor to multiply with dt to get t_end, ensures not too many steps 
    w_0            : Initial step
    fence          : Boolean value if the is a fence

    Asserts:
    - The last element of the time array matches the end time
    '''
    t_end = t_0 + t_end_factor * dt

    def f(t, w, fence, kf, omegaW, FW0):
        # Test function 
        return np.sin(t) + w
     
    t_array, w_matrix = hf.solve_ODE(dt, t_0, t_end, w_0, f, hf.RK4_step, fence, sv.kf, sv.omegaW, sv.FW0)
    assert np.isclose(t_array[-1], t_end)


@given(dt=st.floats(min_value=0.1, max_value=10000000),
       t_0=st.floats(min_value=-10000000, max_value=10000000),
       t_end_factor=st.floats(min_value=1, max_value=10),
       w_0=st.lists(st.floats(min_value=-10000000, max_value=10000000), min_size=2, max_size=2),
       fence = st.booleans())
@settings(max_examples=50)
def test_solve_ODE_stepvalues(dt, t_0, t_end_factor, w_0, fence):
    '''
    Test the solve_ODE function
    Check the solution matrix for each step

    Inputs:
    dt             : Step length
    t_0            : Start time
    t_end          : End time
    t_end_factor   : Factor to multiply with dt to get t_end, ensures not too many steps 
    w_0            : Initial step
    fence          : Boolean value if the is a fence

    Asserts:
    - The solution matrix has the correct values for each step
    '''
    t_end = t_0 + t_end_factor * dt
    def f(t, w, fence, kf, omegaW, FW0):
        # Test function 
        return np.sin(t) + w
     
    t_array, w_matrix = hf.solve_ODE(dt, t_0, t_end, w_0, f, hf.euler_step, fence)

    #num_steps = hf.find_num_steps(t_0, t_end, dt)
    #new_dt = hf.find_actual_dt(t_0, t_end, num_steps)
    new_dt = t_array[1] - t_array[0]

    assert np.allclose(w_matrix[0], w_0), "w_matrix[0] == w_0"

    for i in range(len(t_array) - 1):
        assert np.allclose(w_matrix[i + 1], hf.euler_step(f, t_array[i], w_matrix[i], new_dt, fence))


# Test the two_component_w_RHS

@given(t=st.floats(min_value=0.0, max_value=100.0),
       w=st.lists(st.floats(min_value=-np.pi, max_value=np.pi), min_size=2, max_size=2),
       fence = st.booleans())
def test_two_component_w_RHS_length(t, w,fence):
    '''
    Test for the two_component_w_RHS function.

    Inputs:
    t      : Current time
    w      : Vector [w_0, w_1] representing initial values of theta and omega
    fence  : Boolean value if the is a fence


    Checks:
    - Checks if the output has a length of 2.
    '''

    # Calculate the derivative using the function
    updated_w = hf.two_component_w_RHS(t, w, fence)

    assert len(updated_w) == 2,  "the updated w should have length 2"


@given(t=st.floats(min_value=0.0, max_value=100.0),
       w=st.lists(st.floats(min_value=-np.pi, max_value=np.pi), min_size=2, max_size=2),
       fence = st.booleans())
def test_two_component_w_RHS_theta_update(t, w, fence):
    '''
    Test for the two_component_w_RHS function.

    Inputs:
    t      : Current time
    w      : Vector [w_0, w_1] representing initial values of theta and omega
    fence  : Boolean value if the is a fence

    Checks:
    - Checks if the calculated derivative of theta is equal to the given value of omega.
    '''

    # Calculate the derivative using the function
    updated_w = hf.two_component_w_RHS(t, w, fence)

    assert np.isclose(updated_w[0], w[1])


@given(t=st.floats(min_value=0.0, max_value=100.0),
       w=st.lists(st.floats(min_value=-np.pi, max_value=np.pi), min_size=2, max_size=2),
       fence = st.booleans())
def test_two_component_w_RHS_omega_update(t, w, fence):
    '''
    Test for the two_component_w_RHS function.

    Inputs:
    t      : Current time
    w      : Vector [w_0, w_1] representing initial values of theta and omega
    fence  : Boolean value if the is a fence

    Checks:
    - Checks if the calculated derivative of omega follows the expected formula.
    '''

    # Calculate the derivative using the function
    updated_w = hf.two_component_w_RHS(t, w, fence)

    # Calculate the expected derivative of omega
    expected_w1 = -sv.m * sv.g * sv.h * np.sin(w[0]) / sv.IC

    # Check if the calculated derivative of omega matches the expected value
    assert np.isclose(updated_w[1], expected_w1)


# Test the two_component_w_RHS_small_angle

@given(t=st.floats(min_value=0.0, max_value=100.0),
       w=st.lists(st.floats(min_value=-np.pi, max_value=np.pi), min_size=2, max_size=2),
       fence = st.booleans())
def test_two_component_w_RHS_small_angle_length(t, w, fence):
    '''
    Test for the two_component_w_RHS_small_angle function.

    Inputs:
    t      : Current time
    w      : Vector [w_0, w_1] representing initial values of theta and omega
    fence  : Boolean value if the is a fence

    Checks:
    - The output has a length of 2
    '''

    # Calculate the derivative using the function
    updated_w = hf.two_component_w_RHS_small_angle(t, w, fence)

    assert len(updated_w) == 2,  "the updated w should have length 2"

@given(t=st.floats(min_value=0.0, max_value=100.0),
       w=st.lists(st.floats(min_value=-np.pi, max_value=np.pi), min_size=2, max_size=2),
       fence = st.booleans())
def test_two_component_w_RHS_small_angle_omega_update(t, w, fence):
    '''
    Test for the two_component_w_RHS_small_angle function.

    t      : Current time
    w      : Vector [w_0, w_1] representing initial values of theta and omega
    fence  : Boolean value if the is a fence

    Checks:
    - The calculated derivative of omega follows the expected formula based on the small angle approximation
    '''

    # Calculate the derivative using the function
    updated_w = hf.two_component_w_RHS_small_angle(t, w, fence)

    # Calculate the expected derivative of omega based on the small angle approximation
    expected_w1 = -sv.m * sv.g * sv.h * w[0] / sv.IC

    # Check if the calculated derivative of omega matches the expected value
    assert np.isclose(updated_w[1], expected_w1)

@given(t=st.floats(min_value=0.0, max_value=100.0),
       w=st.lists(st.floats(min_value=-np.pi, max_value=np.pi), min_size=2, max_size=2),
       fence = st.booleans())
def test_two_component_w_RHS_small_angle_theta_update(t, w, fence):
    '''
    Test for the two_component_w_RHS_small_angle function.

    Inputs:
    t      : Current time
    w      : Vector [w_0, w_1] representing initial values of theta and omega
    fence  : Boolean value if the is a fence

    Checks:
    - The calculated derivative of theta is equal to the given value of omega
    '''

    # Calculate the derivative using the function
    updated_w = hf.two_component_w_RHS_small_angle(t, w, fence)

    # Check if the calculated derivative of theta matches the expected value
    assert np.isclose(updated_w[0], w[1]), "updated_w[0] == w[1]"


@given(t_0=st.floats(min_value=0, max_value=10000000),
       t_end_extra=st.floats(min_value=1, max_value=10),
       theta_0=st.floats(min_value=-10000000, max_value=10000000),
       dt_array=st.lists(st.floats(min_value=0.1, max_value=10), min_size=1),
       step_function=st.just(hf.euler_step) | st.just(hf.RK4_step),
       omega_freq=st.floats(0, np.pi))
@settings(max_examples=50)
def test_error_as_func_of_dt_size(t_0, t_end_extra, theta_0, dt_array, step_function, omega_freq):
    """
    Test the error_as_func_of_dt function for the size of the output arrays.

    Inputs:
    t_0            : The initial time of the system.
    t_end_extra    : Additional time to add to t_0 for computing t_end.
    theta_0        : The initial value of the angle (theta).
    dt_array       : An array of different step sizes (dt) to test.
    step_function  : The numerical integration method to use (e.g., Euler's method, RK4).
    omega_freq     : Frequency of the harmonic oscillator.

    Returns:
    None
    """

    t_end = t_0 + t_end_extra

    # Calculate the target value using the analytical solution
    target = hf.analytical_solution_small_angle(t_end, theta_0, omega_freq)

    # Call the function under test
    actual_dt_array, error_array = hf.error_as_func_of_dt(target, dt_array, t_0, t_end, step_function, theta_0)

    # Check that the output arrays have the correct size
    assert len(actual_dt_array) == len(dt_array)
    assert len(error_array) == len(dt_array)

@given(t_0=st.floats(min_value=0, max_value=10000000),
    t_end_extra=st.floats(min_value=1, max_value=10),
    theta_0=st.floats(min_value=-10000000, max_value=10000000),
    dt_array=st.lists(st.floats(min_value=0.1, max_value=10), min_size=1),
    step_function=st.just(hf.euler_step) | st.just(hf.RK4_step),
    omega_freq=st.floats(0, np.pi))
@settings(max_examples=50)
def test_error_as_func_of_dt_values(t_0, t_end_extra, theta_0, dt_array, step_function, omega_freq):
    """
    Test the error_as_func_of_dt function for the values in the output arrays.

    Inputs:
    t_0            : The initial time of the system.
    t_end_extra    : Additional time to add to t_0 for computing t_end.
    theta_0        : The initial value of the angle (theta).
    dt_array       : An array of different step sizes (dt) to test.
    step_function  : The numerical integration method to use (e.g., Euler's method, RK4).
    omega_freq     : Frequency of the harmonic oscillator.

    Returns:
    None
    """

    t_end = t_0 + t_end_extra

    # Calculate the target value using the analytical solution
    target = hf.analytical_solution_small_angle(t_end, theta_0, omega_freq)

    # Call the function under test
    actual_dt_array, error_array = hf.error_as_func_of_dt(target, dt_array, t_0, t_end, step_function, theta_0)

    # Check the values in the output arrays

    # Check that all values in actual_dt_array are floats
    assert all(isinstance(dt, float) for dt in actual_dt_array), "actual_dt_array contains non-float values"

    # Check that all values in error_array are floats
    assert all(isinstance(error, float) for error in error_array), "error_array contains non-float values"

    # Check that all values in actual_dt_array are positive
    assert all(dt > 0 for dt in actual_dt_array), "actual_dt_array contains non-positive values"

    # Check that all values in actual_dt_array and error_array are finite
    assert np.all(np.isfinite(actual_dt_array)), "actual_dt_array contains non-finite values"
    assert np.all(np.isfinite(error_array)), "error_array contains non-finite values"

@given(x=st.floats(min_value=-100, max_value=100),
       a=st.floats(min_value=-100, max_value=100),
       b=st.floats(min_value=-100, max_value=100))
@settings(max_examples=50)
def test_linear_function(x, a, b):
    """
    Test the linear_function for different input values.

    Inputs:
    x : Input value.
    a : Coefficient of the linear term.
    b : Constant term.

    Returns:
    None
    """
    # Call the function under test
    result = hf.linear_function(x, a, b)

    # Calculate the expected result
    expected_result = a * x + b

    # Check if the result matches the expected result
    assert np.isclose(result, expected_result)