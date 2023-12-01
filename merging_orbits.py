import numpy as np

class particle:
    def __init__(self, initial_position, initial_velocity, mass, radius = 10):
        self.position_history = [initial_position]
        self.velocity_history = [initial_velocity]
        self.radius = radius
        self.mass = mass

def radius_to_mass(radius):
  return matter_density * np.pi * (4/3) * (radius**3)

G = 1               # Gravity constant
matter_density = 1  # Density such that: mass_of_particle = 4pi/3 radius^3 * matter_density
N = 8              # Total number of bodies
init_box_length = 1000
particle_list = []

for i in range(N):
  #prepare random parameters
  x = init_box_length*np.random.uniform()
  y = init_box_length*np.random.uniform()
  z = init_box_length*np.random.uniform()
  vx = 0.0005*init_box_length*(np.random.uniform()-0.5)
  vy = 0.0005*init_box_length*(np.random.uniform()-0.5)
  vz = 0.0005*init_box_length*(np.random.uniform()-0.5)
  m = 100*np.random.uniform()
  #create new body
  body = particle(np.array([x, y, z]), np.array([vx,vy,vz]), m)
  particle_list.append(body)

dt = 0.01
simulation_steps = 3000

for t in range(simulation_steps):
  for i in particle_list:
    # Compute total force due to other particles
    for j in particle_list:
      if i != j:
        distance_vector = j.position_history[-1] - i.position_history[-1]
        acceleration_on_i = distance_vector * ((G * j.mass)/(np.linalg.norm(distance_vector)**3))

        velocity = i.velocity_history[-1] + acceleration_on_i * dt
        i.velocity_history.append(velocity)

        position = i.position_history[-1] + velocity * dt
        i.position_history.append(position)

import pandas as pd
import plotly.graph_objects as go

data = []
for p in particle_list:
  x = [v[0] for v in p.position_history]
  y = [v[1] for v in p.position_history]
  z = [v[2] for v in p.position_history]
  data.append(go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode='markers', marker=dict(color='white', size=3)))
  data.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='yellow', width=2.5)))


fig = go.Figure(data)

fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=0),
    paper_bgcolor='black',
    plot_bgcolor='black',
    scene=dict(
        xaxis=dict(showbackground=False, visible=False),
        yaxis=dict(showbackground=False, visible=False),
        zaxis=dict(showbackground=False, visible=False),
    ),
    font=dict(color='white')
)

fig.show()
