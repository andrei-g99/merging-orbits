import numpy as np

class Body:
    def __init__(self, initial_position, initial_velocity, mass, alive_status=True, radius = 10):
        self.position_history = [initial_position]
        self.velocity_history = [initial_velocity]
        self.radius = radius
        self.mass = mass
        self.alive = alive_status

def radius_to_mass(radius):
  return matter_density * np.pi * (4/3) * (radius**3)

G = 1               # Gravity constant
matter_density = 1  # Density such that: mass_of_particle = 4pi/3 radius^3 * matter_density
N = 2              # Total number of bodies
init_box_length = 1000
body_list = []

# for i in range(N):
#   #prepare random parameters
#   x = init_box_length*np.random.uniform()
#   y = init_box_length*np.random.uniform()
#   z = init_box_length*np.random.uniform()
#   vx = 0.0005*init_box_length*(np.random.uniform()-0.5)
#   vy = 0.0005*init_box_length*(np.random.uniform()-0.5)
#   vz = 0.0005*init_box_length*(np.random.uniform()-0.5)
#   m = 100*np.random.uniform()
#   #create new body
#   b = Body(np.array([x, y, z]), np.array([vx, vy, vz]), m, True)
#   body_list.append(b)
body_list.append(Body(np.array([0, 100, 0]), np.array([1, 0, 0]), 10, True))
body_list.append(Body(np.array([0, 0, 0]), np.array([0, 0, 0]), 100, True))
dt = 0.01
simulation_steps = 50000
