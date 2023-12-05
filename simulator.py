import csv
import numpy as np
import os
import config

simulation_data = []
simulation_steps = config.simulation_steps
body_list = config.body_list
init_box_length = config.init_box_length
dt = config.dt
N = config.N
G = config.G

for t in range(simulation_steps):
    for i in body_list:
        # Collision handler and bound check
        if i.alive:
            if np.linalg.norm(i.position_history[-1]) > init_box_length:
                # Body out of sim bounds, eliminate from simulation
                i.alive = False
            else:
                for j in body_list:
                    if i != j and j.alive:
                        distance_vector = j.position_history[-1] - i.position_history[-1]
                        radius_sum = j.radius + i.radius
                        if np.linalg.norm(distance_vector) < radius_sum:
                            # Collision detected between body i and j
                            # The body with smaller mass is eliminated
                            total_mass = i.mass + j.mass

                            velocity_of_merger = ((i.mass)/(total_mass)) * i.velocity_history[-1] + ((j.mass)/(total_mass)) * j.velocity_history[-1]
                            center_of_mass = ((i.mass)/(total_mass)) * i.position_history[-1] + ((j.mass)/(total_mass)) * j.position_history[-1]
                            if i.mass <= j.mass:
                                i.alive = False
                                j.velocity_history[-1] = velocity_of_merger
                                j.position_history[-1] = center_of_mass

                            else:
                                j.alive = False
                                i.velocity_history[-1] = velocity_of_merger
                                i.position_history[-1] = center_of_mass

    for i in body_list:
        # Propagate gravity dynamics
        if i.alive:
            for j in body_list:
                if i != j and j.alive:
                    distance_vector = j.position_history[-1] - i.position_history[-1]
                    acceleration_on_i = distance_vector * ((G * j.mass) / (np.linalg.norm(distance_vector) ** 3))

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
        timestep_data.append(body.alive)

    simulation_data.append(timestep_data)

file_path = f'{config.data_filename}.csv'  # Replace with your file path
try:
    os.remove(file_path)
except FileNotFoundError:
    print('Creating new file')

with open(file_path, 'w', newline='') as file:
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
    print(f'{config.data_filename}.csv has been generated')
