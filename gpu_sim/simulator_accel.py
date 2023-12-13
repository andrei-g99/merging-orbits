import csv
import numpy as np
import math
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import os
import config
from config import mass_to_radius
from tqdm import tqdm

#  Accelerated version of simulator.py
#  Your system must have CUDA Toolkit version 11 installed and a compatible GPU

# GPU kernel code
from kernel_code import kernel_code

# Compile kernel code
mod = SourceModule(kernel_code)
func = mod.get_function("gravitySimulator")

simulation_data = []
simulation_steps = config.simulation_steps
init_box_length = config.init_box_length
dt = config.dt
N = config.N
G = config.G

matter_density = config.matter_density
body_list = config.body_list
threads_per_block = config.threads_per_block

body_dtype = np.dtype([
    ('position', np.float32, 3),
    ('radius', np.float32),
    ('mass', np.float32),
    ('alive', np.int32)
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

# Simulation loop
for t in tqdm(range(simulation_steps), desc='Simulating on GPU'):

    # Prepare input data for GPU
    input_data = structured_array
    output_data = np.zeros_like(input_data)
    input_gpu = cuda.mem_alloc(input_data.nbytes)
    output_gpu = cuda.mem_alloc(output_data.nbytes)

    # Transfer data to GPU
    cuda.memcpy_htod(input_gpu, input_data)

    # Calculate block and grid dimensions
    block_size = (threads_per_block, 1, 1)
    grid_x = math.ceil(len(structured_array) / threads_per_block)
    grid_size = (grid_x, 1, 1)

    # Kernel execution
    func(input_gpu, output_gpu, np.int32(len(structured_array)), block=block_size, grid=grid_size)

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

file_path = f'./../{config.data_filename}_gpu_accel.csv'  # Replace with your file path
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
    print(f'{config.data_filename}_gpu_accel.csv has been generated')
  
