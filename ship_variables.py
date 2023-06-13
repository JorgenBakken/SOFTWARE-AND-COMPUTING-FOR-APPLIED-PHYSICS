import numpy as np
import scipy.constants

import help_functions as hf

g = scipy.constants.g  # Gravitational acceleration (m/s^2)
R = 10  # Ship's radius (m)
h = 4 * R / (3 * np.pi)  # Distance between the metacenter (M) and the ship's center of gravity (C) (m)
sigma0 = 1000  # Mass density of water (kg/m^2)
sigma = 500  # Mass density of the ship (kg/m^2)

A_ship = 0.5 * np.pi * R ** 2  # Cross-sectional area of the ship (m^2)
m = sigma * A_ship  # Mass of the ship (kg)
mL_CONST = 0 * m  # Mass of the load (kg) (constant, set to 0 in this case)

IC = 0.5 * m * R ** 2 * (1 - 32 / (9 * np.pi ** 2))  # Ship's moment of inertia with respect to the axis through the center of mass (kg*m^2)
kf = 100.0  # Friction constant (N*s/m)

T0 = 2 * np.pi * np.sqrt(IC / (m * g * h))  # Period of small harmonic oscillations (s)

FW0 = m * g * 0.625  # Weight of the fluid displaced by the ship in equilibrium (N)
omegaW = 0.93 * 2 * np.pi / T0  # Frequency of small harmonic oscillations in the fluid (rad/s)

A0 = (m + mL_CONST) / sigma0  # Cross-sectional area of the fluid displaced by the ship in equilibrium (m^2)
m0 = sigma0 * A0  # Mass of the fluid displaced by the ship in equilibrium (kg)
FB0 = (m + mL_CONST) * g  # Buoyant force acting on the ship in equilibrium (N)

beta = hf.calculate_beta(sigma, sigma0)  # Sector angle in equilibrium (radians)

A_sector0 = 0.5 * beta * R ** 2  # Area of the sector of the circular segment of water displaced by the ship in equilibrium (m^2)
A_triangle0 = 0.5 * R ** 2 * np.sin(beta)  # Area of the triangular segment of water displaced by the ship in equilibrium (m^2)
d0 = R * (1 - np.cos(beta / 2))  # Depth of the ship's deepest point below the water surface in equilibrium (m)
B0 = R * 4 * (np.sin(beta / 2) ** 3) / (3 * (beta - np.sin(beta)))  # Depth of the buoyancy center below the water surface in equilibrium (m)

yM0 = R * np.cos(beta / 2)  # y-coordinate of the metacenter with respect to the water surface in equilibrium (m)
yD0 = yM0 - R  # y-coordinate of the ship's deepest point with respect to the water surface in equilibrium (m)
yC0 = yM0 - h  # y-coordinate of the ship's center of gravity with respect to the water surface in equilibrium (m)
yB0 = yM0 - B0  # y-coordinate of the buoyancy center with respect to the water surface in equilibrium (m)
yL0 = yM0  # y-coordinate of the load with respect to the water surface in equilibrium (m)
xC0 = 0.0  # x-coordinate of the ship's center of gravity (m)