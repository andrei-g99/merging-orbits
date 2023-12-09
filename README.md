# merging-orbits
Multi-body gravitational simulator where object collisions are handled by a simplified body merger heuristic.

# Setup
- Install Anaconda3 on your system
- Create a new conda virtual environment with Python version 3.10:

(or you can create a new project with PyCharm and select Conda as your interpreter, it will automatically create a new conda env)

(If using windows powershell first run `conda init powershell`, restart the shell and try to activate the env)

`! The current latest compatible version of Python is v3.10 !`
```bash
conda create -n merging_orbits_env python=3.10
conda activate merging_orbits_env
```

- Install all dependencies with PIP:

```bash
pip install vtk
pip install numpy
pip install Pandas
pip install moviepy
pip install mayavi
```

# How to run a simulation
- Set the initial conditions and rendering options in **config.py**
- Run **simulator.py**
- Run **render.py**

# To do list
- Implement Barnes-Hut algorithm with octree
- Implement some 3D spatial partitioning for collision detection
- Tweaks and tests/experiments
- TBD

# Gallery

![Demo](https://github.com/andrei-g99/andrei-g99.github.io/blob/main/mergingorbits.png)
