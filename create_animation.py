import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

import ship_variables as sv

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

