import os.path
from tqdm import tqdm
import numpy as np
from vispy import scene, io
from vispy.visuals.filters import Alpha
import pandas as pd
import meshio
import plotly.graph_objects as go
from matplotlib import pyplot as plt

from pyswcloader.reader import swc




def region_mesh(data_path, color='lightgray', opacity=0.1):
    vertices, faces, normals, _ = io.read_mesh(fname=data_path)
    mesh = scene.visuals.Mesh(vertices=vertices, faces=faces, color=color, shading='smooth')
    mesh.attach(Alpha(opacity))
    return mesh


def neuron_mesh(data_path, color='red', size=1, opacity=1):
    if isinstance(data_path, list):
        neuronlist = data_path
    elif isinstance(data_path, str):
        if ''.join(list(data_path)[-4:]) == '.swc':
            neuronlist = [data_path]
        else:
            neuronlist = swc.read_neuron_path(data_path)
    else:
        raise TypeError('data_path must be str or list-like.')
    coords = pd.DataFrame()
    for _neuron_path in tqdm(neuronlist):
        _coords = swc.read_swc(_neuron_path)
        coords = pd.concat([coords, _coords], axis=0)
    mesh = scene.visuals.Markers(
        pos=np.array(coords[['x', 'y', 'z']]),
        antialias=0,
        edge_width=0,
        scaling=False,
        face_color=color,
        size=size,
        alpha=opacity,
    )
    return mesh


def plot_neuron_3d(neuron_path, region_path=None, bgcolor=(1, 1, 1, 0), **kwargs):
    canvas = scene.SceneCanvas(bgcolor=bgcolor, keys='interactive', show=False)
    view = canvas.central_widget.add_view()
    neurons = neuron_mesh(data_path=neuron_path, **kwargs)
    view.add(neurons)
    if region_path != None:
        regions = region_mesh(data_path=region_path)
        view.add(regions)
    camera = scene.cameras.TurntableCamera()
    camera.elevation = -90
    camera.azimuth = 0
    camera.up = '+z'
    view.camera = camera
    return canvas


def plot_neuron_2d_vispy(neuron_path, region_path=None, bgcolor=(1, 1, 1, 0), perspective='sagittal', region_color='gray',
                   region_opacity=0.05, show=True, save_path=None, **kwargs):
    if perspective in ['coronal', 'sagittal', 'horizontal']:
        canvas = scene.SceneCanvas(bgcolor=bgcolor, keys='interactive', show=False)
        view = canvas.central_widget.add_view()
        neurons = neuron_mesh(data_path=neuron_path, **kwargs)
        view.add(neurons)
        if region_path and os.path.exists(region_path):
            regions = region_mesh(data_path=region_path, color=region_color, opacity=region_opacity)
            view.add(regions)

        camera = scene.cameras.TurntableCamera()
        if perspective == 'sagittal':
            camera.elevation = -90
            camera.azimuth = 0
            camera.up = '+z'
        elif perspective == 'coronal':
            camera.center = (13200 / 2, 8000 / 2, 11400 / 2)
            camera.up = '-y'
            camera.elevation = 0
            camera.azimuth = -1.5
            camera.fov = 0
        elif perspective == 'horizontal':
            camera.center = (13200 / 2, 8000 / 2, 11400 / 2)
            camera.elevation = 2
            camera.azimuth = 0
            camera.fov = 0
        view.camera = camera
        img = canvas.render()

        if show == True:
            plt.imshow(img)
            plt.axis('off')

        if save_path != None:
            io.image.imsave(filename=save_path, im=img)
    else:
        raise ValueError('perspective value must be coronal, sagittal or horizontal.')

    return img


def read_neuron(data_path, color='red'):
    if isinstance(data_path, list):
        neuronlist = data_path
    elif isinstance(data_path, str):
        if ''.join(list(data_path)[-4:]) == '.swc':
            neuronlist = [data_path]
        else:
            neuronlist = swc.read_neuron_path(data_path)
    else:
        raise TypeError('data_path must be str or list-like.')
    plots = []
    for _neuron_path in tqdm(neuronlist):
        neu = swc.read_swc(_neuron_path)
        __coords = neu[['x', 'y', 'z']].values
        plots.append(go.Scatter3d(x=__coords[:, 0], y=__coords[:, 1], z=__coords[:, 2], mode='markers',
                                   marker=dict(size=0.3, color=color), showlegend=False))
    return plots


def plot_neuron_2d_plotly(neuron_path, region_path, perspective='sagittal', color='red', bgcolor='rgba(0,0,0,0)',
                              region_color='gray', region_opacity=0.05, show=False, save_path=None):
    if perspective in ['coronal', 'sagittal', 'horizontal']:
        plots = read_neuron(neuron_path, color)

        if os.path.exists(region_path):
            vol = meshio.read(region_path)
            verts = vol.points
            I, J, K = vol.cells_dict["triangle"].T
            x, y, z = verts.T
            plots.append(go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, color=region_color, opacity=region_opacity))
        fig = go.Figure(data=plots)
        if perspective == 'sagittal':
            view = dict(up=dict(x=0, y=-1, z=0), eye=dict(x=0, y=0, z=-1.25))
        elif perspective == 'horizontal':
            view = dict(eye=dict(x=0, y=-1.25, z=0))
        else:
            view = dict(up=dict(x=0, y=-1, z=0), eye=dict(x=1.25, y=0, z=0))
        fig.update_layout(
            paper_bgcolor=bgcolor,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            scene_camera=view
        )
        if show:
            fig.show()
        if save_path is not None:
            fig.write_image(save_path)
        return





