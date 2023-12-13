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

# Compile kernel code
mod = SourceModule(kernel_code)
func = mod.get_function("gravitySimulator")

simulation_data = []

body_dtype = np.dtype([
    ('position', np.float32, 3),
    ('radius', np.float32),
    ('mass', np.float32),
    ('alive', np.int32)
    ('accel_due_to', np.float32, (N, 3))
])

structured_array = np.empty(len(body_list), dtype=body_dtype)

# Initialize bodies
for i, body in enumerate(body_list):
    structured_array[i]['position'] = body.position_history[-1]
    structured_array[i]['mass'] = body.mass
    structured_array[i]['radius'] = body.radius
    if body.alive:
        structured_array[i]['alive'] = int(1)
    else:
        structured_array[i]['alive'] = int(0)
    
    for k in range(N):
        structured_array[i]['accel_due_to'][k][0] = 0
        structured_array[i]['accel_due_to'][k][1] = 0
        structured_array[i]['accel_due_to'][k][2] = 0


# Simulation loop
for t in tqdm(range(simulation_steps), desc='Simulating on GPU'):

    # Prepare input data for GPU
    input_data = structured_array
    output_body_data = np.zeros_like(input_data)
    input_gpu = cuda.mem_alloc(input_data.nbytes)
    output_body_gpu = cuda.mem_alloc(output_body_data.nbytes)

    # Transfer data to GPU
    cuda.memcpy_htod(input_gpu, input_data)

    # Calculate block and grid dimensions
    block_size = (math.sqrt(threads_per_block), math.sqrt(threads_per_block), 1)
    grid_x = math.ceil(len(structured_array) / threads_per_block)
    grid_size = (grid_x, 1, 1)

    # Kernel execution
    func(input_gpu, output_body_gpu, np.int32(len(structured_array)), block=block_size, grid=grid_size)

    # Wait for kernel to finish
    cuda.Context.synchronize()

    # Retrieve
    cuda.memcpy_dtoh(output_data, output_gpu)

    print(output_data)

    # Free GPU memory
    input_gpu.free()
    output_gpu.free()

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

file_path = f'./../{data_filename}_gpu_accel.csv'  # Replace with your file path
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
    print(f'{data_filename}_gpu_accel.csv has been generated')