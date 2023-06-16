import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

import ship_variables as sv

#---------------------------------------------------------------------------------------------------------------------------

def init_anim():
    """ 
    Initializes the animation.
    """

    # Define global variables for the plot elements
    global ax, boat, deck, load, CM, left_fence, right_fence, textbox_theory
    
    # Create plot elements
    boat, = plt.plot([], [], color="k", linewidth=1)  # The boat plot element
    deck, = plt.plot([], [], color="k", linewidth=1)  # The deck plot element
    sea_surface, = plt.plot([-sv.R*10, sv.R*10], [0, 0], color='blue', linewidth=2)  # The surface plot element
    load, = plt.plot([], [], color="r", marker="o", markersize=10)  # The load plot element
    CM, = plt.plot([], [], color="g", marker="o", markersize=10)  # The center of mass plot element
    left_fence, = plt.plot([], [], color="k", marker="|", markersize=25)  # The left fence plot element
    right_fence, = plt.plot([], [], color="k", marker="|", markersize=25)  # The right fence plot element
    
    # Set plot properties
    ax.set_xlim([-sv.R*1.3, sv.R*1.3])  # Set the x-axis limits
    ax.set_ylim([-sv.R*1.1, sv.R*1.1])  # Set the y-axis limits
    ax.set_xlabel('$x$')  # Set the x-axis label
    ax.set_ylabel('$y$')  # Set the y-axis label
    ax.set_aspect("equal")  # Set the aspect ratio of the plot
    ax.set_title('Animation')  # Set the plot title
    
    # Create a textbox for displaying theory information
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textbox_theory = ax.text(0.775, 0.95, '', transform=ax.transAxes, fontsize=12,
                             verticalalignment='top', bbox=props)  # The textbox for theory information

    # Return the initialized plot elements
    return ax, boat, deck, load, CM, left_fence, right_fence, textbox_theory

#---------------------------------------------------------------------------------------------------------------------------

def animate(M, theta, t, xC, yC, sL, fence=False):
    '''
    Updates the animation for each frame.
    
    Parameters:
        M (int): Current frame index
        theta (array): Array of angle values for each frame
        t (array): Array of time values for each frame
        xC (array): Array of x-coordinate values for the center of mass for each frame
        yC (array): Array of y-coordinate values for the center of mass for each frame
        sL (array): Array of load position values relative to the center of mass for each frame
        fence (bool): Flag indicating whether to display the fence
        
    Returns:
        Tuple of updated plot elements
    '''
    global ax, boat, deck, load, CM, left_fence, right_fence, textbox_theory
    
    # Set the x-axis and y-axis limits
    ax.set_xlim([-sv.R * 1.1 + np.amin(xC), sv.R * 1.1 + np.amax(xC)])
    ax.set_ylim([-sv.R * 1.1, sv.R * 1.1])
    
    # Calculate the angle values for the boat shape
    angle_values = np.linspace(0, np.pi, 100)
    
    # Calculate the position of the metasenter
    metasenter_x = xC[M] - sv.h * np.sin(theta[M])
    metasenter_y = yC[M] + sv.h * np.cos(theta[M])
    
    # Calculate the x and y coordinates of the boat shape
    xs = sv.R * np.cos(angle_values + np.pi + theta[M]) + metasenter_x
    ys = sv.R * np.sin(angle_values + np.pi + theta[M]) + metasenter_y
    
    # Update the boat and deck plot elements
    boat.set_data(xs, ys)
    deck.set_data([xs[0], xs[-1]], [ys[0], ys[-1]])
    
    # Update the load plot element if it is present
    if sL[M] != -42:
        load.set_data(metasenter_x + sL[M] * np.cos(theta[M]), metasenter_y + sL[M] * np.sin(theta[M]))
    
    # Update the center of mass plot element
    CM.set_data(xC[M], yC[M])
    
    # Update the fence plot elements if the fence flag is True
    if fence:
        left_fence.set_data([metasenter_x - sv.R * np.cos(theta[M])], [metasenter_y - sv.R * np.sin(theta[M])])
        right_fence.set_data([metasenter_x + sv.R * np.cos(theta[M])], [metasenter_y + sv.R * np.sin(theta[M])])
    
    # Set the text for the theory information
    theta_string = r'$\theta = %.2f$' % (theta[M] * 180 / np.pi) + r"$\degree$"
    time_string = '$t =  %.2f$' % (t[M])
    textbox_theory.set_text(theta_string + '\n' + time_string)
    
    # Increment the frame index
    M += 1
    
    # Return the updated plot elements
    return ax, boat, deck, load, CM, left_fence, right_fence, textbox_theory

#---------------------------------------------------------------------------------------------------------------------------

def animate_deck_movement(t, theta, x_C, y_C, s_L=[], fence=False, stepsize=0.01, show_axis_values=False):
    """
    Creates an animation that shows the dynamics of the ship's deck movement.

    Parameters:
        t (array): Array of time values for which the system's \vec{w} has been calculated
        theta (array): Array of the ship's angular displacements
        x_C (array): Array of x-coordinate values for the center of mass
        y_C (array): Array of y-coordinate values for the center of mass
        s_L (array, optional): Array of the load's position relative to the center of mass
        fence (bool, optional): Flag indicating whether to draw fences on the ship
        stepsize (float, optional): Time interval between each frame
        show_axis_values (bool, optional): If True, axis values will be visible, making the animation slightly less smooth

    Returns:
        Animation showing the dynamics of the ship's movement.
    """
    global fig, ax
    fig, ax = plt.subplots()

    # Calculate the time step
    dt = t[1] - t[0]

    # Determine the number of frames to skip based on the desired step size
    skips = max(int(stepsize / dt), 1)

    # Extract the subset of data for animation
    theta_anim = theta[::skips]
    t_anim = t[::skips]
    x_C_anim = x_C[::skips]
    y_C_anim = y_C[::skips]

    # If load positions are not provided, use a default value
    if len(s_L) == 0:
        s_L_anim = -42 * np.ones(len(theta_anim))
    else:
        s_L_anim = s_L[::skips]

    # Create the animation using the animate function
    h_anim = animation.FuncAnimation(fig, animate, init_func=init_anim, frames=len(t_anim) - 1, interval=1,
                                     blit=not show_axis_values,
                                     fargs=(theta_anim, t_anim, x_C_anim, y_C_anim, s_L_anim, fence))
    
    # Display the animation
    plt.show()
