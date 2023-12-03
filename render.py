import numpy as np
import mayavi.mlab as mlab
import vtk
import pandas as pd
from moviepy.editor import ImageSequenceClip


data = pd.read_csv('simulation_data.csv')

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)

renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

points = vtk.vtkPoints()
# Add your 3D points here
# Example: points.InsertNextPoint(1.0, 2.0, 3.0)
for index, row in data.iterrows():
    N = row['number_of_particles']
    timestep = row['timestep']
    for i in range(int(N)):
        x = float(row[f'particle_{i}_pos_x'])
        y = float(row[f'particle_{i}_pos_y'])
        z = float(row[f'particle_{i}_pos_z'])
        m = float(row[f'particle_{i}_mass'])

        points.InsertNextPoint(x, y, z)
    # Access data from each row with row['column_name']
    # For example, row['name'] or row['age'] if your columns are named 'name' and 'age'


polydata = vtk.vtkPolyData()
polydata.SetPoints(points)

glyphFilter = vtk.vtkGlyph3D()
glyphFilter.SetInputData(polydata)
glyphFilter.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(glyphFilter.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer.AddActor(actor)

renderer.SetBackground(0, 0, 0)  # RGB background color
renderWindow.SetSize(800, 600)  # Window size

renderWindow.Render()
renderWindowInteractor.Start()
