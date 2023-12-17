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
    ('alive', np.int32)
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
# for i, body in enumerate(body_array):
#     body_array[i]['position'][0] = init_box_length*(np.random.uniform()-0.5)
#     body_array[i]['position'][1] = init_box_length*(np.random.uniform()-0.5)
#     body_array[i]['position'][2] = init_box_length*(np.random.uniform()-0.5)

#     body_array[i]['velocity'][0] = 5*(np.random.uniform()-0.5)
#     body_array[i]['velocity'][1] = 5*(np.random.uniform()-0.5)
#     body_array[i]['velocity'][2] = 5*(np.random.uniform()-0.5)

#     r = 10*(np.random.uniform()) + 10

#     body_array[i]['mass'] = radius_to_mass(r)
#     body_array[i]['radius'] = r
#     body_array[i]['alive'] = int(1)

# Manual initialization
body_array[0]['position'][0] = 0
body_array[0]['position'][1] = 300
body_array[0]['position'][2] = 0
body_array[0]['velocity'][0] = 0
body_array[0]['velocity'][1] = 0
body_array[0]['velocity'][2] = 0
r = 10
body_array[0]['mass'] = radius_to_mass(r)
body_array[0]['radius'] = r
body_array[0]['alive'] = int(1)

body_array[1]['position'][0] = 0
body_array[1]['position'][1] = 0
body_array[1]['position'][2] = 0
body_array[1]['velocity'][0] = 0
body_array[1]['velocity'][1] = 0
body_array[1]['velocity'][2] = 0
r = 30
body_array[1]['mass'] = radius_to_mass(r)
body_array[1]['radius'] = r
body_array[1]['alive'] = int(1)


# Calculate block and grid dimensions
block_size = (threads_per_block, 1, 1)
grid_x = math.ceil(len(body_array) / threads_per_block)
grid_size = (grid_x, 1, 1)

alive_cnt = np.array([N], dtype=np.int32)

alive_cnt_gpu = cuda.mem_alloc(alive_cnt.nbytes)

input_gpu = cuda.mem_alloc(body_array.nbytes)
output_gpu = cuda.mem_alloc(body_array.nbytes)

# Simulation loop
for t in tqdm(range(simulation_steps), desc='Simulating on GPU'):

    output_data = np.zeros_like(body_array)
    # Transfer data to GPU
    cuda.memcpy_htod(input_gpu, body_array)
    cuda.memcpy_htod(alive_cnt_gpu, alive_cnt)

    # Kernel execution
    func(input_gpu, output_gpu, alive_cnt_gpu, np.int32(len(body_array)), np.float32(G), np.float32(dt), np.float32(init_box_length), block=block_size, grid=grid_size)

    # Wait for kernel to finish
    cuda.Context.synchronize()

    # Retrieve
    cuda.memcpy_dtoh(output_data, output_gpu)
    cuda.memcpy_dtoh(alive_cnt, alive_cnt_gpu)

    body_array = output_data

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
        timestep_data.append(body_array[i]['mass'])
        timestep_data.append(body_array[i]['alive'])

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
    for b in range(N):
        header += [f'body_{b}_pos_x', f'body_{b}_pos_y', f'body_{b}_pos_z',
                   f'body_{b}_vel_x', f'body_{b}_vel_y', f'body_{b}_vel_z', f'body_{b}_radius', f'body_{b}_mass', f'body_{b}_alive']
    writer.writerow(header)


    # Write the data
    for row in tqdm(simulation_data, desc='Saving simulation data'):
        writer.writerow(row)
    print(f'{data_filename}.csv has been generated')

# Free GPU memory
input_gpu.free()
output_gpu.free()