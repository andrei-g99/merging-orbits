# merging-orbits
Multi-body gravitational simulator where object collisions are handled by a simplified body merger heuristic.

# Setup
- Install Anaconda3 on your system
- Create a new conda virtual environment with Python version 3.10:

(or you can create a new project with PyCharm and select Conda as your interpreter, it will automatically create a new conda env)

(If using windows powershell first run `conda init powershell`, restart the shell and try to activate the env)

```bash
conda create -n condaenv python=3.10
conda activate condaenv
```

- Install all dependencies with PIP:

```bash
pip install vtk
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

# Gallery

![Demo](https://github.com/andrei-g99/andrei-g99.github.io/blob/main/mergingorbits.png)
