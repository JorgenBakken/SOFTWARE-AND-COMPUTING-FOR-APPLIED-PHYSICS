This project focuses on studying the dynamics and stability characteristics of a ship model. The model used is a long, semicircular ship rocking about an axis in the longitudinal direction. Various factors influencing the ship's behavior are considered, including its external forces, friction constants, and the presence of a moving load on the deck.

## Table on contents 

- [Introduction](#introduction)
- [Ship Model](#ship-model)
- [External Forces on the Ship](#external-forces-on-the-ship)
- [Harmonic Oscillations](#harmonic-oscillations)
- [Moving Load](#moving-load)
- [Equations of Motion](#equations-of-motion)
- [Getting Started](#getting-started)

<!----><a name="introduction"></a>
### Introduction
The Vasa, a Swedish warship, serves as inspiration for this project. It sank during its inaugural voyage in 1628 and was later raised in 1961. The ship's instability and subsequent sinking provide a real-world case study to explore the dynamics and stability of ships.

<!----><a name="ship-model"></a>
### Ship Model
The ship model used in this project is a long, semicircular ship. It can be either compact, with a constant mass density, or hollow, with a thin hull and deck. The center of mass of the ship, denoted as C, is located at a specific distance below the midpoint of the deck. The volume of water displaced by the ship varies as it rocks, and the dynamics are described using sector angles and displaced water areas.

<!----><a name="external-forces-on-the-ship"></a>
### External Forces on the Ship
In addition to gravity, the ship experiences external forces such as buoyancy, friction, and wind. The buoyant force, acting at the center of gravity of the displaced water, counteracts the ship's weight. Frictional forces between the ship and water contribute to damping effects, while wind forces act horizontally.

<!----><a name="harmonic-oscillations"></a>
### Harmonic Oscillations 
When considering only gravity and buoyancy as external forces, the ship behaves as a harmonic oscillator. Its motion is characterized by small oscillations around the equilibrium position. The ship's angular frequency, period, and damping are derived based on its physical properties.

<!----><a name="moving-load"></a>
### Moving Load 
The presence of a moving load on the ship's deck introduces additional dynamics and affects stability. The load is modeled as a point mass initially at rest. Its motion is governed by gravity and the normal force from the deck. Frictional effects between the load and deck can also be included in the model.

<!----><a name="equations-of-motion"></a>
### Equations of Motion
To describe the ship's motion, Newton's second law is applied to the center of mass and rotational motion. The equations of motion consider the forces acting on the ship, including gravity, buoyancy, friction, wind, and the contact forces from the moving load. These equations provide insights into the ship's behavior and stability.

<!----><a name="getting-started"></a>
### Getting Started
To get started with this project, follow these steps:

Clone the repository to your local machine.
Ensure you have the required dependencies and libraries installed.
Open the project in your preferred programming environment.

Usage

Once the project is set up, you can explore different scenarios and parameters to analyze the ship's dynamics and stability. Adjust the ship's properties, external forces, and the presence of a moving load to observe their effects on the system. Analyze the resulting motion, oscillations, and stability characteristics. To get more background info about the project, and more detailed physics, read the Vasa_ship.pdf.

