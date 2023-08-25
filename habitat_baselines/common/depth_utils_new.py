# Code borrowed from https://github.com/s-gupta/map-plan-baseline/tree/master
# and https://github.com/devendrachaplot/Object-Goal-Navigation
# ==============================================================================

"""Utilities for processing depth images.
"""
import numpy as np
import habitat_baselines.common.rotation_utils as ru
from argparse import Namespace 
import torch

def get_camera_matrix(width, height, fov):
  """Returns a camera matrix from image size and fov."""
  xc = (width-1.) / 2.
  zc = (height-1.) / 2.
  f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
  camera_matrix = {'xc':xc, 'zc':zc, 'f':f}
  camera_matrix = Namespace(**camera_matrix)
  return camera_matrix

def get_point_cloud_from_z(Y, camera_matrix, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    x, z = np.meshgrid(np.arange(Y.shape[-1]),
                       np.arange(Y.shape[-2] - 1, -1, -1))
    for _ in range(Y.ndim - 2):
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)
    X = (x[::scale, ::scale] - camera_matrix.xc) * \
        Y[::scale, ::scale] / camera_matrix.f
    Z = (z[::scale, ::scale] - camera_matrix.zc) * \
        Y[::scale, ::scale] / camera_matrix.f
    XYZ = np.concatenate((X[..., np.newaxis],
                          Y[::scale, ::scale][..., np.newaxis],
                          Z[..., np.newaxis]), axis=X.ndim)
    return XYZ


def transform_camera_view(XYZ, sensor_height, camera_elevation_degree):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix(
        [1., 0., 0.], angle=np.deg2rad(camera_elevation_degree))
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ

def transform_pose(XYZ, location, theta):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    sh = XYZ.shape
    R = ru.get_r_matrix([0., 0., 1.], angle=theta)
    XYZ = np.matmul(XYZ.reshape(sh[0], sh[1]*sh[2], 3), np.transpose(R, axes=(0,2,1))).reshape(XYZ.shape)
    XYZ[:, :, :, 0] = XYZ[:, :, :, 0] + location[:, 1][..., np.newaxis, np.newaxis]
    XYZ[:, :, :, 1] = XYZ[:, :, :, 1] + location[:, 0][..., np.newaxis, np.newaxis]
    return XYZ

def bin_points_w_sem(XYZS_cms, map_size, z_bins, xy_resolution, map_center):
    """Bins points into xy-z bins
    XYZS_cms is ... x H x W x4 -> 3D coordinates + semantic channel
    Outputs is ... x map_size x map_size x (len(z_bins)+1)
    """
    sh = XYZS_cms.shape
    XYZS_cms = XYZS_cms.reshape([-1, sh[-3], sh[-2], sh[-1]])
    n_z_bins = len(z_bins) + 1
    
    counts = []
    sem_labels = []
    for i, XYZS_cm in enumerate(XYZS_cms):
        isnotnan = np.logical_not(np.isnan(XYZS_cm[:, :, 0]))
        X_bin = np.round((XYZS_cm[:, :, 0] / xy_resolution) + map_center[0]).astype(np.int32)
        Y_bin = np.round((XYZS_cm[:, :, 1] / xy_resolution) + map_center[1]).astype(np.int32)
        Z_bin = np.digitize(XYZS_cm[:, :, 2], bins=z_bins).astype(np.int32)
        S_bin = XYZS_cm[:, :, 3].astype(np.int32)
        
        isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0,
                        Y_bin < map_size,
                        Z_bin >= 0, Z_bin < n_z_bins, isnotnan, S_bin > 0])
        isvalid = np.all(isvalid, axis=0)

        ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin
        ind[np.logical_not(isvalid)] = 0

        count = np.bincount(ind[i].ravel(), isvalid[i].ravel().astype(np.int32),
                            minlength=map_size * map_size * n_z_bins)
        
        sem_label = count.copy()
        sem_label[ind[i][isvalid[i]]] = S_bin[i][isvalid[i]]    #  replace count with the semantic label
        
        counts.append(np.reshape(count, [map_size, map_size, n_z_bins]))
        sem_labels.append(np.reshape(sem_label, [map_size, map_size, n_z_bins]))

    counts = np.concatenate(counts, axis=0).reshape(list(sh[:-3]) + [map_size, map_size, n_z_bins])
    sem_labels = np.concatenate(sem_labels, axis=0).reshape(list(sh[:-3]) + [map_size, map_size, n_z_bins])

    return counts, sem_labels

def project_sem_w_scores(XYZS_cms, map_size, z_bins, xy_resolution, map_center):
    """Bins points into xy-z bins
    XYZS_cms is ... x H x W x4 -> 3D coordinates + semantic channel
    Outputs is ... x map_size x map_size x (len(z_bins)+1)
    """
    sh = XYZS_cms.shape
    XYZS_cms = XYZS_cms.reshape([-1, sh[-3], sh[-2], sh[-1]])
    n_z_bins = len(z_bins) + 1
    
    counts = []
    sem_labels = []
    sem_scores = []
    for i, XYZS_cm in enumerate(XYZS_cms):
        isnotnan = np.logical_not(np.isnan(XYZS_cm[:, :, 0]))
        X_bin = np.round((XYZS_cm[:, :, 0] / xy_resolution) + map_center[0]).astype(np.int32)
        Y_bin = np.round((XYZS_cm[:, :, 1] / xy_resolution) + map_center[1]).astype(np.int32)
        Z_bin = np.digitize(XYZS_cm[:, :, 2], bins=z_bins).astype(np.int32)
        S_bin = XYZS_cm[:, :, 3].astype(np.int32)
        S_score_bin = XYZS_cm[:, :, 4].astype(np.int32)
        
        isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0,
                        Y_bin < map_size,
                        Z_bin >= 0, Z_bin < n_z_bins, isnotnan, S_bin > 0])
        isvalid = np.all(isvalid, axis=0)

        ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin
        ind[np.logical_not(isvalid)] = 0

        count = np.bincount(ind[i].ravel(), isvalid[i].ravel().astype(np.int32),
                            minlength=map_size * map_size * n_z_bins)
        
        sem_label = count.copy()
        sem_label[ind[i][isvalid[i]]] = S_bin[i][isvalid[i]]    #  replace count with the semantic label
        sem_score = np.zeros((map_size * map_size * n_z_bins))
        sem_score[ind[i]]=S_score_bin[i]
        
        counts.append(np.reshape(count, [map_size, map_size, n_z_bins]))
        sem_labels.append(np.reshape(sem_label, [map_size, map_size, n_z_bins]))
        sem_scores.append(np.reshape(sem_score, [map_size, map_size, n_z_bins]))

    counts = np.concatenate(counts, axis=0).reshape(list(sh[:-3]) + [map_size, map_size, n_z_bins])
    sem_labels = np.concatenate(sem_labels, axis=0).reshape(list(sh[:-3]) + [map_size, map_size, n_z_bins])
    sem_scores = np.concatenate(sem_scores, axis=0).reshape(list(sh[:-3]) + [map_size, map_size, n_z_bins])

    return sem_labels, sem_scores, counts

def get_point_cloud_from_z_t(Y_t, camera_matrix, device, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    grid_x, grid_z = torch.meshgrid(torch.arange(Y_t.shape[-1]),
                                    torch.arange(Y_t.shape[-2] - 1, -1, -1))
    grid_x = grid_x.transpose(1, 0).to(device)
    grid_z = grid_z.transpose(1, 0).to(device)
    grid_x = grid_x.unsqueeze(0).expand(Y_t.size())
    grid_z = grid_z.unsqueeze(0).expand(Y_t.size())

    X_t = (grid_x[:, ::scale, ::scale] - camera_matrix.xc) * \
        Y_t[:, ::scale, ::scale] / camera_matrix.f
    Z_t = (grid_z[:, ::scale, ::scale] - camera_matrix.zc) * \
        Y_t[:, ::scale, ::scale] / camera_matrix.f

    XYZ = torch.stack(
        (X_t, Y_t[:, ::scale, ::scale], Z_t), dim=len(Y_t.size()))

    return XYZ

def transform_camera_view_t(
        XYZ, sensor_height, camera_elevation_degree, device):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix(
        [1., 0., 0.], angle=np.deg2rad(camera_elevation_degree))
    XYZ = torch.matmul(XYZ.reshape(-1, 3),
                       torch.from_numpy(R).float().transpose(1, 0).to(device)
                       ).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ

def transform_pose_t(XYZ, current_pose, device):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([0., 0., 1.], angle=current_pose[2] - np.pi / 2.)
    XYZ = torch.matmul(XYZ.reshape(-1, 3),
                       torch.from_numpy(R).float().transpose(1, 0).to(device)
                       ).reshape(XYZ.shape)
    XYZ[..., 0] += current_pose[0]
    XYZ[..., 1] += current_pose[1]
    return XYZ
