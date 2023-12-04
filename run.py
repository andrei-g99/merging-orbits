import csv
import numpy as np
import os

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
N = 100              # Total number of bodies
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
  body = particle(np.array([x, y, z]), np.array([vx, vy, vz]), m)
  particle_list.append(body)

dt = 0.1
simulation_steps = 10000
simulation_data = []

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

    # Collecting data at the current timestep
    timestep_data = []
    timestep_data.append(t)
    timestep_data.append(N)
    for particle in particle_list:
        #timestep_data.append(particle.position_history[-1].tolist() + particle.velocity_history[-1].tolist() + [particle.mass])
        timestep_data.append(particle.position_history[-1][0])
        timestep_data.append(particle.position_history[-1][1])
        timestep_data.append(particle.position_history[-1][2])
        timestep_data.append(particle.velocity_history[-1][0])
        timestep_data.append(particle.velocity_history[-1][1])
        timestep_data.append(particle.velocity_history[-1][2])
        timestep_data.append(particle.mass)

    simulation_data.append(timestep_data)

file_path = 'simulation_data.csv'  # Replace with your file path
try:
    os.remove(file_path)
except:
    print('Creating new file')

with open('simulation_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write a header row (optional, depending on your data structure)
    header = ['timestep', 'number_of_particles']
    for i in range(len(particle_list)):
        header += [f'particle_{i}_pos_x', f'particle_{i}_pos_y', f'particle_{i}_pos_z',
                   f'particle_{i}_vel_x', f'particle_{i}_vel_y', f'particle_{i}_vel_z', f'particle_{i}_mass']
    writer.writerow(header)

    # Write the data
    for row in simulation_data:
        writer.writerow(row)
