import help_functions as hf

import numpy as np

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


