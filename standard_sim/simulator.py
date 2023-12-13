import csv
import numpy as np
import os
import config
from config import mass_to_radius
from tqdm import tqdm

simulation_data = []
simulation_steps = config.simulation_steps
body_list = config.body_list
init_box_length = config.init_box_length
dt = config.dt
N = config.N
G = config.G
matter_density = config.matter_density


def bodiesAlive():
    alive_index_list = []
    for i in range(len(body_list)):
        if body_list[i].alive:
            alive_index_list.append(i)
    return alive_index_list


def collisionsLoop():
    for k in body_list:
        # Collision handler and bound check
        if k.alive:
            if np.linalg.norm(k.position_history[-1]) > 10*init_box_length:
                # Body out of sim bounds, eliminate from simulation
                k.alive = False
            else:
                for j in body_list:
                    if k != j and j.alive:
                        distance_vector = j.position_history[-1] - k.position_history[-1]
                        radius_sum = j.radius + k.radius
                        if np.linalg.norm(distance_vector) < radius_sum:
                            # Collision detected between body i and j
                            # The body with smaller mass is eliminated
                            total_mass = k.mass + j.mass
                            new_radius = mass_to_radius(total_mass)
                            velocity_of_merger = (k.mass / total_mass) * k.velocity_history[-1] + (j.mass / total_mass) * j.velocity_history[-1]
                            center_of_mass = (k.mass / total_mass) * k.position_history[-1] + (j.mass / total_mass) * j.position_history[-1]
                            if k.mass <= j.mass:
                                k.alive = False
                                j.velocity_history[-1] = velocity_of_merger
                                j.position_history[-1] = center_of_mass
                                j.mass = total_mass
                                j.radius = new_radius

                            else:
                                j.alive = False
                                k.velocity_history[-1] = velocity_of_merger
                                k.position_history[-1] = center_of_mass
                                k.mass = total_mass
                                k.radius = new_radius

def gravityLoop():
    alive_index_list = bodiesAlive()
    if len(alive_index_list) > 1:
        for i in body_list:
            
            total_accel = 0
            # Propagate gravity dynamics
            if i.alive:
                for j in body_list:
                    if i != j and j.alive:
                        distance_vector = j.position_history[-1] - i.position_history[-1]
                        acceleration_ij = distance_vector * ((G * j.mass) / (np.linalg.norm(distance_vector) ** 3))
                        total_accel += acceleration_ij
        
            velocity = i.velocity_history[-1] + total_accel * dt
            i.velocity_history.append(velocity)

            position = i.position_history[-1] + velocity * dt
            i.position_history.append(position)
        
    else:
        if len(alive_index_list) > 0:
            position = body_list[alive_index_list[0]].position_history[-1] + \
                       body_list[alive_index_list[0]].velocity_history[-1] * dt
            body_list[alive_index_list[0]].position_history.append(position)

# Simulation loop
for t in tqdm(range(simulation_steps), desc='Simulating'):

    collisionsLoop()
    gravityLoop()

    # Collecting data at the current timestep
    timestep_data = []
    timestep_data.append(t)
    timestep_data.append(N)
    for body in body_list:
        timestep_data.append(body.position_history[-1][0])
        timestep_data.append(body.position_history[-1][1])
        timestep_data.append(body.position_history[-1][2])
        timestep_data.append(body.velocity_history[-1][0])
        timestep_data.append(body.velocity_history[-1][1])
        timestep_data.append(body.velocity_history[-1][2])
        timestep_data.append(body.radius)
        timestep_data.append(body.alive)

    simulation_data.append(timestep_data)

file_path = f'./../{config.data_filename}.csv'  # Replace with your file path
try:
    os.remove(file_path)
except FileNotFoundError:
    print('Creating new file')

with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write a header row (optional, depending on your data structure)
    header = ['timestep', 'number_of_bodies']
    for b in range(len(body_list)):
        header += [f'body_{b}_pos_x', f'body_{b}_pos_y', f'body_{b}_pos_z',
                   f'body_{b}_vel_x', f'body_{b}_vel_y', f'body_{b}_vel_z', f'body_{b}_radius', f'body_{b}_status']
    writer.writerow(header)

    # Write the data
    for row in tqdm(simulation_data, desc='Saving simulation data'):
        writer.writerow(row)
    print(f'{config.data_filename}.csv has been generated')
