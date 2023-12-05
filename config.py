import numpy as np

# SIM CONFIGS
class Body:
    def __init__(self, initial_position, initial_velocity, mass, alive_status=True, radius=10):
        self.position_history = [initial_position]
        self.velocity_history = [initial_velocity]
        self.radius = radius
        self.mass = mass
        self.alive = alive_status


G = 0.2               # Gravity constant
matter_density = 1  # Density such that: mass_of_particle = 4pi/3 radius^3 * matter_density
init_box_length = 1000
body_list = []

def radius_to_mass(radius):
  return matter_density * np.pi * (4/3) * (radius**3)


n = 10
for i in range(n):
    # prepare random parameters
    x = init_box_length*np.random.uniform()
    y = init_box_length*np.random.uniform()
    z = init_box_length*np.random.uniform()
    vx = 0.5*(np.random.uniform()-0.5)
    vy = 0.5*(np.random.uniform()-0.5)
    vz = 0.5*(np.random.uniform()-0.5)
    r = 10*(np.random.uniform()) + 10
    m = radius_to_mass(r)
    # create new body
    b = Body(np.array([x, y, z]), np.array([vx, vy, vz]), m, True, r)
    body_list.append(b)

# Uncomment below for a 2-body orbiting demo
# body_list.append(Body(np.array([0, 100, 0]), np.array([1, 0, 0]), 10, True))
# body_list.append(Body(np.array([0, 0, 0]), np.array([0, 0, 0]), 100, True))

N = len(body_list)  # Total number of bodies
dt = 0.01
simulation_steps = 50000

data_filename = 'sim_data'

# END SIM CONFIGS

# RENDERING CONFIGS

FPS = 60
video_duration = 10 # Seconds
camera_position = [-1000, 0, 250] # Camera absolute position
camera_direction = [0, 0, 0]  # Pointing direction (focal point)
spheres_default_radius = 1
sphere_color = [1, 1, 1]  # RGB spheres color
background_color = [0, 0, 0]  # RGB background color
window_resolution = [1920, 1080]  # Pixel resolution width x height

video_filename = 'simrun'
# END RENDER CONFIGS
