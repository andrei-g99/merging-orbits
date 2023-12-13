import numpy as np

# SIM CONFIGS


class Body:
    def __init__(self, initial_position, initial_velocity, mass, alive_status=True, radius=10):
        self.position_history = [initial_position]
        self.velocity_history = [initial_velocity]
        self.radius = radius
        self.mass = mass
        self.alive = alive_status


G = 0.1               # Gravity constant
matter_density = 1  # Density such that: mass_of_particle = 4pi/3 radius^3 * matter_density
init_box_length = 1000
body_list = []


def radius_to_mass(radius):
    return matter_density * np.pi * (4/3) * (radius**3)


def mass_to_radius(m):
    return ((m/matter_density) * (3/(4*np.pi)))**(1/3)


n = 30
for i in range(n-1):
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

N = len(body_list)  # Total number of bodies
dt = 0.01
simulation_steps = 2500

data_filename = 'sim_data'

# END SIM CONFIGS

# GPU Acceleration settings - Do NOT touch unless you need to adapt it to your GPU architecture
threads_per_block = 256