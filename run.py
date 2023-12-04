import csv
import numpy as np
import os

class Body:
    def __init__(self, initial_position, initial_velocity, mass, alive_status=True, radius = 10):
        self.position_history = [initial_position]
        self.velocity_history = [initial_velocity]
        self.radius = radius
        self.mass = mass
        self.alive = alive_status

def radius_to_mass(radius):
  return matter_density * np.pi * (4/3) * (radius**3)

G = 1               # Gravity constant
matter_density = 1  # Density such that: mass_of_particle = 4pi/3 radius^3 * matter_density
N = 10              # Total number of bodies
init_box_length = 1000
body_list = []

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
  b = Body(np.array([x, y, z]), np.array([vx, vy, vz]), m, True)
  body_list.append(b)

dt = 0.01
simulation_steps = 5000
simulation_data = []

for t in range(simulation_steps):
    for i in body_list:
        # Compute total force due to other particles
        for j in body_list:
            if i != j:
                distance_vector = j.position_history[-1] - i.position_history[-1]
                acceleration_on_i = distance_vector * ((G * j.mass)/(np.linalg.norm(distance_vector)**3))

                velocity = i.velocity_history[-1] + acceleration_on_i * dt
                i.velocity_history.append(velocity)

                position = i.position_history[-1] + velocity * dt
                i.position_history.append(position)

    # Collecting data at the current timestep
    timestep_data = []
    timestep_data.append(t)
    timestep_data.append(N)
    for body in body_list:
        #timestep_data.append(particle.position_history[-1].tolist() + particle.velocity_history[-1].tolist() + [particle.mass])
        timestep_data.append(body.position_history[-1][0])
        timestep_data.append(body.position_history[-1][1])
        timestep_data.append(body.position_history[-1][2])
        timestep_data.append(body.velocity_history[-1][0])
        timestep_data.append(body.velocity_history[-1][1])
        timestep_data.append(body.velocity_history[-1][2])
        timestep_data.append(body.mass)
        timestep_data.append(1)

    simulation_data.append(timestep_data)

file_path = 'simulation_data.csv'  # Replace with your file path
try:
    os.remove(file_path)
except FileNotFoundError:
    print('Creating new file')

with open('simulation_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write a header row (optional, depending on your data structure)
    header = ['timestep', 'number_of_bodies']
    for i in range(len(body_list)):
        header += [f'body_{i}_pos_x', f'body_{i}_pos_y', f'body_{i}_pos_z',
                   f'body_{i}_vel_x', f'body_{i}_vel_y', f'body_{i}_vel_z', f'body_{i}_mass', f'body_{i}_status']
    writer.writerow(header)

    # Write the data
    for row in simulation_data:
        writer.writerow(row)
    print('simulation_data.csv has been generated')
