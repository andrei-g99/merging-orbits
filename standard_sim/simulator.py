import csv
import numpy as np
import os
from tqdm import tqdm
import json
import numpy as np

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

matter_density = config['simulation']['matter_density'];
G = config['simulation']['gravity_constant'];
init_box_length = config['simulation']['init_box_length'];
matter_density = config['simulation']['matter_density'];
N = config['simulation']['number_of_bodies'];
simulation_steps = config['simulation']['simulation_steps'];
dt = config['simulation']['sizeof_timestep'];
data_filename = config['simulation']['data_filename'];

# SIM CONFIGS


class Body:
    def __init__(self, initial_position, initial_velocity, mass, alive_status=True, radius=10):
        self.position_history = [initial_position]
        self.velocity_history = [initial_velocity]
        self.radius = radius
        self.mass = mass
        self.alive = alive_status


body_list = []


def radius_to_mass(radius):
    return matter_density * np.pi * (4/3) * (radius**3)


def mass_to_radius(m):
    return ((m/matter_density) * (3/(4*np.pi)))**(1/3)


for i in range(N-1):
    # prepare random parameters
    x = init_box_length*(np.random.uniform()-0.5)
    y = init_box_length*(np.random.uniform()-0.5)
    z = init_box_length*(np.random.uniform()-0.5)
    vx = 5*(np.random.uniform()-0.5)
    vy = 5*(np.random.uniform()-0.5)
    vz = 5*(np.random.uniform()-0.5)
    r = 10*(np.random.uniform()) + 10
    m = radius_to_mass(r)
    # create new body
    b = Body(np.array([x, y, z]), np.array([vx, vy, vz]), m, True, r)
    body_list.append(b)

sun = Body(np.array([0, 0, 0]), np.array([0, 0, 0]), radius_to_mass(60), True, 60)
body_list.append(sun)
# Uncomment below for a 2-body orbiting demo
# body_list.append(Body(np.array([0, 100, 0]), np.array([1, 0, 0]), 10, True))
# body_list.append(Body(np.array([0, 0, 0]), np.array([0, 0, 0]), 100, True))


# END SIM CONFIGS


simulation_data = []

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

file_path = f'{data_filename}.csv'  # Replace with your file path
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
    print(f'{data_filename}.csv has been generated')