# merging-orbits
Multi-body gravitational simulator where object collisions are handled by a simplified body merger heuristic.

# Dependencies
- Conda environment
- vtk -> PyQt5 -> mayavi
- Pandas
- moviepy
- numpy
- csv

# To do list
- Create collision handler
- Collision handler should look for bodies in space cells with a count of 2 or more bodies occupying them
    - This avoids O(N^2) check for collisions over all bodies
- Tweaks and tests/experiments
- TBD
