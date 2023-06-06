
# Equations of Motion

To describe the motion/dynamics of the ship, I will use Newton's second law for translational and rotational motion. The equation that describes the translational motion is given by:

$$
\sum \mathbf{F} = m \mathbf{A} = m \frac{d\mathbf{V}}{dt} = \frac{d^2\mathbf{R}}{dt^2}
$$

where $\mathbf{A}$, $\mathbf{V}$, and $\mathbf{R}$ are the acceleration, velocity, and position of the ship's center of mass, respectively. In component form, we can write $\mathbf{R} = (x_C, y_C)$. The rotational motion is given by:

$$
\sum \tau = I_C \frac{d\omega}{dt}
$$

where $\tau$ represents the torque and $I_C$ is the moment of inertia about the axis passing through the center of mass.

I will add a moving load on the ship's deck, whcih will result in a contact force from the load on the ship, contributing to both $\mathbf{F}$ and $\tau$. The forces acting on the ship are:

$$
F_G = -mg \quad \text{(Gravitational force in the y-direction)}
$$

$$
F_B = A\sigma_0g \quad \text{(Boyancy force in the y-direction)}
$$

$$
f = -k_fR\gamma\omega \quad \text{(Friction force in the x-direction)}
$$

$$
F_w = F_0\cos(\omega_wt) \quad \text{(Wind force in the x-direction)}
$$

$$
F_L^y = -m_Lg\cos^2\theta \quad \text{(Load force in the y-direction)}
$$

$$
F_L^x = m_Lg\cos\theta\sin\theta \quad \text{(Load forece in the x-direction)}
$$

The torques acting on the ship will vary both in terms of the magnitude of the force and the distance between the center of mass and the extension of the force ("arm"). For simplicity, we provide all the expressions for the different torques acting on the ship with respect to the axis through C:

$$
\tau_B = -F_B h \sin\theta  \quad \text{(Boyancy torque)}
$$

$$
\tau_f = f(y_C - (R(\cos(\gamma/2)) - 1)) \quad \text{(Frictional torque)}
$$

$$
\tau_w = F_w y_C \quad \text{(Wind torque)}
$$

$$
\tau_L = -m_L g \cos\theta s_L \quad \text{(Lift torque)}
$$

In this project, I will solve the coupled system of differential equations given by the equations in this section. This will be done step by step, gradually adding the different forces.
