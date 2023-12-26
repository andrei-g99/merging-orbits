# merging-orbits
N-body gravitational simulator where object collisions are handled by a simplified body merger heuristic.

Now with GPU accelerated version. (CUDA)

# Basic setup
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

# Optional setup for the GPU accelerated version only

For the GPU accelerated version you must have an NVIDIA GPU with CUDA API version 11 support. [NVIDIA CUDA Toolkit DOWNLOAD](https://developer.nvidia.com/cuda-toolkit)
- Download version 11 for your architecture
- Reboot your system
- Activate the conda env again and install the pycuda python library
```
pip install pycuda
```

( if you don't have Visual Studio installed you need to install it for the NVCC compiler and cl tool - install with the C/C++ toolkit - and remember to add the cl.exe directory C:\Program Files\Microsoft Visual Studio\2022\<yourversion>\VC\Tools\MSVC\<yourversion>\bin\Hostx64\x64 to PATH )

- For the accelerated version run the `simulator_accel.py` script

# How to run a simulation
- Set the initial conditions and rendering options in **config.json**
- Run **simulator.py** (**simulator_accel.py** for the GPU accelerated version)
- Run **render.py**

# To do list
- [ ] Implement Barnes-Hut algorithm with octree
- [ ] Implement a 3D spatial partitioning strategy to speed up the collision handling/detection
- [x] GPU Acceleration with CUDA

# Gallery

![Demo](https://github.com/andrei-g99/andrei-g99.github.io/blob/main/mergingorbits.png)
