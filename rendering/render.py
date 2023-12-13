import os
import config
import numpy as np
import mayavi.mlab as mlab
import math
import glob
import vtk
import pandas as pd
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
from vtkmodules.vtkIOImage import (
    vtkBMPWriter,
    vtkJPEGWriter,
    vtkPNGWriter,
    vtkPNMWriter,
    vtkPostScriptWriter,
    vtkTIFFWriter
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkWindowToImageFilter
)

def WriteImage(fileName, renWin, rgba=True):
    '''
    Write the render window view to an image file.

    Image types supported are:
     BMP, JPEG, PNM, PNG, PostScript, TIFF.
    The default parameters are used for all writers, change as needed.

    :param fileName: The file name, if no extension then PNG is assumed.
    :param renWin: The render window.
    :param rgba: Used to set the buffer type.
    :return:
    '''

    import os

    if fileName:
        # Select the writer to use.
        path, ext = os.path.splitext(fileName)
        ext = ext.lower()
        if not ext:
            ext = '.png'
            fileName = fileName + ext
        if ext == '.bmp':
            writer = vtkBMPWriter()
        elif ext == '.jpg':
            writer = vtkJPEGWriter()
        elif ext == '.pnm':
            writer = vtkPNMWriter()
        elif ext == '.ps':
            if rgba:
                rgba = False
            writer = vtkPostScriptWriter()
        elif ext == '.tiff':
            writer = vtkTIFFWriter()
        else:
            writer = vtkPNGWriter()

        windowto_image_filter = vtkWindowToImageFilter()
        windowto_image_filter.SetInput(renWin)
        windowto_image_filter.SetScale(1)  # image quality
        if rgba:
            windowto_image_filter.SetInputBufferTypeToRGBA()
        else:
            windowto_image_filter.SetInputBufferTypeToRGB()
            # Read from the front buffer.
            windowto_image_filter.ReadFrontBufferOff()
            windowto_image_filter.Update()

        writer.SetFileName(fileName)
        writer.SetInputConnection(windowto_image_filter.GetOutputPort())
        writer.Write()
    else:
        raise RuntimeError('Need a filename.')


data = pd.read_csv(f'{config.data_filename}.csv')

# Controlling FPS and time: FPS * video_duration(seconds) = sim_timesteps
# e.g. FPS = 60 and video_duration = 10
# sim_timesteps = 600, we need to sample only 600 timesteps regardless of how many the simulation provides!

FPS = config.FPS
video_duration = config.video_duration
steps_to_sample = FPS*video_duration
total_timesteps = len(data['timestep'])

if steps_to_sample > total_timesteps:
    steps_to_sample = total_timesteps

sampling_cnt_threshold = math.floor(total_timesteps/steps_to_sample)

# Initialize vtk renderer etc.
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)

renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Set camera
# Comment this section for default behaviour

camera = vtk.vtkCamera()
camera.SetPosition(config.camera_position[0], config.camera_position[1], config.camera_position[2])
camera.SetFocalPoint(config.camera_direction[0], config.camera_direction[1], config.camera_direction[2])
camera.SetClippingRange(config.nearclip, config.farclip)
renderer.SetActiveCamera(camera)

# Sphere source for glyphs
sphereSource = vtk.vtkSphereSource()
sphereSource.SetRadius(config.spheres_default_radius)  # Set the radius of the spheres

frames = []

# Add 3D points
# Example: points.InsertNextPoint(1.0, 2.0, 3.0)
cnt = 1
for index, row in tqdm(data.iterrows(), desc='Rendering'):
    if cnt == sampling_cnt_threshold:
        cnt = 1
        points = vtk.vtkPoints()
        N = row['number_of_bodies']
        timestep = row['timestep']

        scalars = vtk.vtkFloatArray()
        for i in range(int(N)):
            if row[f'body_{i}_status']:
                x = float(row[f'body_{i}_pos_x'])
                y = float(row[f'body_{i}_pos_y'])
                z = float(row[f'body_{i}_pos_z'])
                r = float(row[f'body_{i}_radius'])
                scalars.InsertNextValue(r)
                points.InsertNextPoint(x, y, z)
        # Access data from each row with row['column_name']
        # For example, row['name'] or row['age'] if your columns are named 'name' and 'age'

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().SetScalars(scalars)

        glyphFilter = vtk.vtkGlyph3D()
        glyphFilter.SetSourceConnection(sphereSource.GetOutputPort())
        glyphFilter.SetInputData(polydata)
        glyphFilter.SetScaleModeToScaleByScalar()
        glyphFilter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyphFilter.GetOutputPort())
        mapper.SetScalarVisibility(False)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(config.sphere_color[0], config.sphere_color[1], config.sphere_color[2])
        renderer.AddActor(actor)

        renderer.SetBackground(config.background_color[0], config.background_color[1], config.background_color[2])  # RGB background color
        renderWindow.SetSize(config.window_resolution[0], config.window_resolution[1])  # Window size
        # Render and save frame
        renderWindow.Render()
        frame_filename = f"frame_{timestep}.png"
        WriteImage(frame_filename, renderWindow, rgba=True)
        frames.append(frame_filename)

        # Remove actor to prepare for next timestep
        renderer.RemoveActor(actor)

    cnt += 1

file_path = f'{config.video_filename}.mp4'  # Replace with your file path
try:
    os.remove(file_path)
except FileNotFoundError:
    print('Creating new file')

# Compile frames into a video
clip = ImageSequenceClip(frames, fps=FPS)  # fps can be adjusted
clip.write_videofile(f'{config.video_filename}.mp4', codec="libx264")
renderWindowInteractor.Start()

# Path where the files are located (assuming current directory for this example)
path = '.'

# Pattern to match files of the format 'frame_{number}.png'
pattern = os.path.join(path, 'frame_*')

# Find all files matching the pattern
files_to_delete = glob.glob(pattern)

# Delete the files
for file in files_to_delete:
    os.remove(file)
