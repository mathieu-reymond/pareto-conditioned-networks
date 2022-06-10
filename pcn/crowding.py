import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def crowding_distance(points):
    # first normalize accross dimensions
    points = (points-points.min(axis=0))/(points.ptp(axis=0)+1e-8)
    # sort points per dimension
    dim_sorted = np.argsort(points, axis=0)
    point_sorted = np.take_along_axis(points, dim_sorted, axis=0)
    # compute distances between lower and higher point
    distances = np.abs(point_sorted[:-2] - point_sorted[2:])
    # pad extrema's with 1, for each dimension
    distances = np.pad(distances, ((1,), (0,)), constant_values=1)
    # sum distances of each dimension of the same point
    crowding = np.zeros(points.shape)
    crowding[dim_sorted, np.arange(points.shape[-1])] = distances
    crowding = np.sum(crowding, axis=-1)
    return crowding


def max_crowding_increase(points):
    # for each point, what is the impact of removing it wrt total crowding distance
    distances = [crowding_distance(np.concatenate((points[:i], points[i+1:]), axis=0)).sum() for i in range(len(points))]
    distances = np.array(distances)
    return distances


def plot3d_points(points, step=0, threshold=0.):
    crowding = crowding_distance(points)
    big = np.argwhere(crowding >= threshold).flatten()
    crowding = crowding[big]
    points = points[big]

    x, y, z = points[:,0], points[:,1], points[:,2]

    colors = cm.hsv(crowding/np.amax(crowding))
    colmap = cm.ScalarMappable(cmap=cm.hsv)
    colmap.set_array(crowding)

    # Plot the surface.
    #fig = plt.figure()
    fig = plt.gcf()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, c=colors, marker='o')
    plt.title(f'step {step}')
    ax = fig.add_subplot(111)
    ax.scatter(x, y, c=colors, marker='o')
    cb = fig.colorbar(colmap)
    return fig


def plot2d_points(points, threshold=0.):
    crowding = crowding_distance(points)
    big = np.argwhere(crowding >= threshold).flatten()
    crowding = crowding[big]
    points = points[big]

    x, y = points[:,0], points[:,1]

    colors = cm.hsv(crowding/np.amax(crowding))
    colmap = cm.ScalarMappable(cmap=cm.hsv)
    colmap.set_array(crowding)

    # Plot the surface.
    # fig = plt.figure()
    fig = plt.gcf()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, c=colors, marker='o')
    cb = fig.colorbar(colmap)
    return fig


if __name__ == '__main__':
    import pandas as pd

    front = pd.read_csv('minecart.csv', header=None).to_numpy()
    plot3d_points(front, threshold=.3)
