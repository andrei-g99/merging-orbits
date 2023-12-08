# merging-orbits
Multi-body gravitational simulator where object collisions are handled by a simplified body merger heuristic.

# Setup
- Create a new Anaconda3 virtual environment with Python version 3.9

```bash
conda create -n mayavi_env python=3.9
conda activate mayavi_env
```

- Install VTK:

```bash
  conda install -c conda-forge vtk
```

- Install all dependencies with PIP:

```bash
pip install mayavi
pip install PyQt5
pip install numpy
pip install Pandas
pip install moviepy
```

# How to run a simulation
- Set the initial conditions and rendering options in **config.py**
- Run **simulator.py**
- Run **render.py**

# To do list
- Create collision handler
- Collision handler should look for bodies in space cells with a count of 2 or more bodies occupying them
    - This avoids O(N^2) check for collisions over all bodies
- Tweaks and tests/experiments
- TBD
