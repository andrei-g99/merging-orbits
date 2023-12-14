import csv
import numpy as np
import math
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import os
import json
from tqdm import tqdm



#  Accelerated version of simulator.py
#  Your system must have CUDA Toolkit version 11 installed and a compatible GPU

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
threads_per_block = config['acceleration_settings']['threads_per_block'];

# GPU kernel code
from kernel_code import kernel_code



class Body:
    def __init__(self, initial_position, initial_velocity, mass, alive_status=True, radius=10):
        self.position_history = [initial_position]
        self.velocity_history = [initial_velocity]
        self.radius = radius
        self.mass = mass
        self.alive = alive_status



def radius_to_mass(radius):
    return matter_density * np.pi * (4/3) * (radius**3)


def mass_to_radius(m):
    return ((m/matter_density) * (3/(4*np.pi)))**(1/3)


body_dtype = np.dtype([
    ('position', np.float32, 3),
    ('velocity', np.float32, 3),
    ('radius', np.float32),
    ('mass', np.float32),
    ('alive', np.int32),
    ('accel_due_to', np.float32, (N, 3))
])

body_array = np.empty(N, dtype=body_dtype)



# for i in range(N-1):
#     # prepare random parameters
#     x = init_box_length*(np.random.uniform()-0.5)
#     y = init_box_length*(np.random.uniform()-0.5)
#     z = init_box_length*(np.random.uniform()-0.5)
#     vx = 5*(np.random.uniform()-0.5)
#     vy = 5*(np.random.uniform()-0.5)
#     vz = 5*(np.random.uniform()-0.5)
#     r = 10*(np.random.uniform()) + 10
#     m = radius_to_mass(r)
#     # create new body
#     b = Body(np.array([x, y, z]), np.array([vx, vy, vz]), m, True, r)
#     body_list.append(b)

# sun = Body(np.array([0, 0, 0]), np.array([0, 0, 0]), radius_to_mass(60), True, 60)
# body_list.append(sun)
# Uncomment below for a 2-body orbiting demo
# body_list.append(Body(np.array([0, 100, 0]), np.array([1, 0, 0]), 10, True))
# body_list.append(Body(np.array([0, 0, 0]), np.array([0, 0, 0]), 100, True))

# Compile kernel code
mod = SourceModule(kernel_code)
func = mod.get_function("gravitySimulator")

simulation_data = []

# Initialize bodies
for i, body in enumerate(body_array):
    body_array[i]['position'][0] = init_box_length*(np.random.uniform()-0.5)
    body_array[i]['position'][1] = init_box_length*(np.random.uniform()-0.5)
    body_array[i]['position'][2] = init_box_length*(np.random.uniform()-0.5)

    body_array[i]['velocity'][0] = 5*(np.random.uniform()-0.5)
    body_array[i]['velocity'][1] = 5*(np.random.uniform()-0.5)
    body_array[i]['velocity'][2] = 5*(np.random.uniform()-0.5)

    r = 10*(np.random.uniform()) + 10

    body_array[i]['mass'] = radius_to_mass(r)
    body_array[i]['radius'] = r
    body_array[i]['alive'] = int(1)

    for k in range(N):
        body_array[i]['accel_due_to'][k][0] = 0
        body_array[i]['accel_due_to'][k][1] = 0
        body_array[i]['accel_due_to'][k][2] = 0

# Calculate block and grid dimensions
block_size = (int(math.sqrt(threads_per_block)), int(math.sqrt(threads_per_block)), 1)
grid_x = math.ceil(len(body_array) / threads_per_block)
grid_size = (grid_x, 1, 1)

input_gpu = cuda.mem_alloc(body_array.nbytes)
output_gpu = cuda.mem_alloc(body_array.nbytes)

# Simulation loop
for t in tqdm(range(simulation_steps), desc='Simulating on GPU'):
    at_least_two_alive_flag = 0
    for i in body_array:
        if i['alive'] == int(1):
            at_least_two_alive_flag += 1
            if at_least_two_alive_flag == 2:
                at_least_two_alive_flag = 1
                break

    if at_least_two_alive_flag == 1:
            output_data = np.zeros_like(body_array)

            # Transfer data to GPU
            cuda.memcpy_htod(input_gpu, body_array)
        
            # Kernel execution
            func(input_gpu, output_gpu, np.int32(len(body_array)), block=block_size, grid=grid_size)
        
            # Wait for kernel to finish
            cuda.Context.synchronize()
        
            # Retrieve
            cuda.memcpy_dtoh(output_data, output_gpu)
            
        
            for i in range(len(output_data)):
                total_accel = np.array([0, 0, 0], np.float32)
                for k in range(N):
                    total_accel += np.array([output_data[i]['accel_due_to'][k][0], output_data[i]['accel_due_to'][k][1], output_data[i]['accel_due_to'][k][2]], np.float32)
                
                body_array[i]['position'] = output_data[i]['position']
                body_array[i]['velocity'] = output_data[i]['velocity']
                body_array[i]['mass'] = output_data[i]['mass']
                body_array[i]['radius'] = output_data[i]['radius']
                body_array[i]['alive'] = output_data[i]['alive']
        
        
                body_array[i]['velocity'] = body_array[i]['velocity'] + total_accel * dt
        
        
                body_array[i]['position'] = body_array[i]['position'] + body_array[i]['velocity'] * dt
        
            # Collecting data at the current timestep
            timestep_data = []
            timestep_data.append(t)
            timestep_data.append(N)
            for i in range(len(body_array)):
                timestep_data.append(body_array[i]['position'][0])
                timestep_data.append(body_array[i]['position'][1])
                timestep_data.append(body_array[i]['position'][2])
                timestep_data.append(body_array[i]['velocity'][0])
                timestep_data.append(body_array[i]['velocity'][1])
                timestep_data.append(body_array[i]['velocity'][2])
                timestep_data.append(body_array[i]['radius'])
                timestep_data.append(body_array[i]['alive'])
        
            simulation_data.append(timestep_data)


file_path = f'{data_filename}_gpu_accel.csv'  # Replace with your file path
try:
    os.remove(file_path)
except FileNotFoundError:
    print('Creating new file')

with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write a header row (optional, depending on your data structure)
    header = ['timestep', 'number_of_bodies']
    for b in range(N):
        header += [f'body_{b}_pos_x', f'body_{b}_pos_y', f'body_{b}_pos_z',
                   f'body_{b}_vel_x', f'body_{b}_vel_y', f'body_{b}_vel_z', f'body_{b}_radius', f'body_{b}_mass', f'body_{b}_alive']
    writer.writerow(header)


    # Write the data
    for row in tqdm(simulation_data, desc='Saving simulation data'):
        writer.writerow(row)
    print(f'{data_filename}_gpu_accel.csv has been generated')

# Free GPU memory
input_gpu.free()
output_gpu.free()