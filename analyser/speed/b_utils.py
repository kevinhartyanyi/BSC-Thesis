import math
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import png
import skimage.measure 
#import skimage.segmentation


width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351
width_to_focal[1226] = 707.0493     # TODO check correct focal length

fps = 10
baseline = 0.54
min_depth = 1e-3
max_depth = 80
max_velocity = 60


def listDirectory(dir_name, extension=None):
    """
    List all files with the given extension in the specified directory.
    :param dir_name: name of the directory
    :param extension: filename suffix
    :return: list of file locations  
    """
    if extension is None:
        is_valid = lambda fn: os.path.isfile(fn)
    else:
        is_valid = lambda fn: fn[-len(extension):] == extension
    
    fns = [os.path.join(dir_name, fn) 
           for fn in os.listdir(dir_name) if is_valid(os.path.join(dir_name, fn))]
    fns.sort()
    return fns


def read_gslic_labels(label_fn, n_sps=-1):
    """
    Read GSLICr labels from .pgm file
    :param label_fn: name of the label file
    :param n_sps: number of superpixels
    :return: superpixel labels in matrix
    """
    assert os.path.isfile(label_fn)
    assert os.path.splitext(label_fn)[1] == '.pgm'
    labels = cv2.imread(label_fn, cv2.IMREAD_UNCHANGED)
    assert n_sps == -1 or n_sps >= len(np.unique(labels))
    return labels


def read_boruvka_labels(label_fn, n_sps=-1):
    """
    Read Boruvka labels
    :param label_fn: name of the label file
    :param n_sps: number of superpixels
    :return: superpixel labels in matrix
    """
    assert os.path.isfile(label_fn)
    labels = cv2.imread(label_fn, -1)
    assert n_sps == -1 or n_sps == len(np.unique(labels))
    return labels


def readDepth(depth_fn, width, height):
    """
    Read disparity map and converts it to depth map
    :param depth_fn: name of the depth file
    :param width: width of the original image
    :param height: height of the original image
    :return: depth values (in meter) in matrix
    """
    disparity = np.load(depth_fn)
    disparity = cv2.resize(disparity, (width, height), interpolation=cv2.INTER_LINEAR)
    depth = width_to_focal[width] * baseline / (width * disparity)
    depth[depth > max_depth] = max_depth
    depth[depth < min_depth] = min_depth
    return depth


def read_gt_depth(depth_fn, width, height):
    """
    Read ground truth disparity map and converts it to depth map
    :param depth_fn: name of the depth file
    :param width: width of the original image
    :param height: height of the original image
    :return: depth values (in meter) in matrix
    """
    disparity = cv2.imread(depth_fn, -1)
    disparity = cv2.resize(disparity, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = (disparity > 0)
    disparity = disparity.astype(np.float32) / 256
    depth = width_to_focal[disparity.shape[1]] * baseline / (disparity + (1. - mask))
    mask = np.logical_or(depth <= min_depth, depth >= max_depth)
    depth[mask] = 0
    return depth


def read_gt_flow(flow_fn):
    """
    Read from KITTI .png file
    :param flow_fn: name of the flow file
    :return: optical flow data in matrix
    """
    flow_object = png.Reader(filename=flow_fn)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']

    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return flow[:, :, :2]


def average(data, labels, leave_out=None):
    """
    Calculate the averages value of each superpixel 
    :param data: data matrix
    :param labels: superpixel labels
    :return: average superpixel values in matrix
    """
    sp_ids = np.unique(labels)
    avg = np.zeros_like(data)
    for sp_id in sp_ids:
        avg[labels == sp_id] = np.mean(data[labels == sp_id])
    return avg


def average_gt(data, labels):
    """
    Calculate the averages ground truth value of each superpixel 
    :param data: data matrix, 0 in case of invalid values
    :param labels: superpixel labels
    :return: average superpixel values in matrix
    """
    sp_ids = np.unique(labels)
    avg = np.zeros_like(data)
    masked_labels = np.copy(labels)
    masked_labels[data == 0] = 99999
    sp_ids = np.unique(masked_labels)
    for sp_id in sp_ids:
        if sp_id == 99999:
            continue
        avg[labels == sp_id] = np.mean(data[masked_labels == sp_id])
    return avg


def calcAngleOfView(focal_length, width, height):
    """
    Calculate the horizontal and veritcal angle of view for a given region
    :param focal_length: focal length of the image
    :param width: width of the region
    :param height: height of the region
    :return: horizontal and veritcal angle of view
    """
    horizontal_angle = 2 * math.degrees(math.tan((width / 2) / focal_length))
    vertical_angle = 2 * math.degrees(math.tan((height / 2) / focal_length))
    return horizontal_angle, vertical_angle


def calcSize(h_angle, v_angle, depth):
    """
    Calculate the size of a given region
    :param h_angle: horizontal angle of view
    :param v_angle: veritcal angle of view
    :param depth: distance in meters
    :return: width and height in meters
    """
    h_angle = math.radians(h_angle / 2)
    width = math.atan(h_angle) * depth * 2

    v_angle = math.radians(v_angle / 2)
    height = math.atan(v_angle) * depth * 2

    return width, height


def normalize(v):
    """
    Normalize the given vector
    :param v: float vector
    :return: normalized vector
    """
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

"""
def drawVelocityVectors(labels, velocity, img_fn, draw_boundaries=True):
    
    Draw velocity vectors on the given image
    :param labels: superpixel labels
    :param velocity: velocity matrix
    :param img_fn: name of the image file
    :param draw_boundaries: whether to draw superpixel boundaries or not
    :return: image with velocity vectors
    
    regions = skimage.measure.regionprops(labels)

    # Compute the maximum l (longest flow)
    l_max = 0.
    for props in regions:
        cx, cy = props.centroid
        cx, cy = int(cx), int(cy)
        dx, dy, dz = velocity[cx, cy, :]
        l = math.sqrt(dx*dx + dy*dy)
        if l > l_max: 
            l_max = l

    img = cv2.imread(img_fn, cv2.IMREAD_COLOR)  # height x width x channels
    
    # Draw boundaries
    if draw_boundaries:
        boundaries = skimage.segmentation.find_boundaries(labels)
        img[boundaries] = 255

    # Draw arrows
    height, width, _ = velocity.shape
    for props in regions:
        cx, cy = props.centroid
        cx, cy = int(cx), int(cy)
        dx, dy, dz = velocity[cx, cy]
        l = math.sqrt(dx*dx + dy*dy)
        
        if l > 0:
            # Factor to normalise the size of the spin depeding on the length of the arrow
            spin_size = 5.0 * l / l_max
            nx, ny = int(cx + dx), int(cy + dy)

            # print((nx, ny))
            nx = min(max(0, nx), height - 1) 
            ny = min(max(0, ny), width - 1)

            cx, cy = cy, cx
            nx, ny = ny, nx

            cv2.line(img, (cx, cy), (nx, ny), (0, 255, 0), 1, cv2.LINE_AA)

            # Draws the spin of the arrow
            angle = math.atan2(cy - ny, cx - nx)

            cx = int(nx + spin_size * math.cos(angle + math.pi / 4))
            cy = int(ny + spin_size * math.sin(angle + math.pi / 4))
            cv2.line(img, (cx, cy), (nx, ny), (0, 255, 0), 1, cv2.LINE_AA, 0)
            
            cx = int(nx + spin_size * math.cos(angle - math.pi / 4))
            cy = int(ny + spin_size * math.sin(angle - math.pi / 4))
            cv2.line(img, (cx, cy), (nx, ny), (0, 255, 0), 1, cv2.LINE_AA, 0)
    return img
"""
def calculateShiftedLabels(labels, avg_flow):
    """
    Shift the given superpixel labels with the average optical flow 
    :param labels: superpixel labels
    :param avg_flow: average optical flow matrix
    :return: shifted superpixel labels
    """
    height, width = labels.shape
    shifted_labels = np.ones(labels.shape, dtype=np.int16) * -1
    for sp_id in np.unique(labels):
        w_sp_x, w_sp_y = np.where(labels == sp_id)
        of = avg_flow[w_sp_x[0], w_sp_y[0]]

        sp_x = w_sp_x + int(of[1])
        sp_y = w_sp_y + int(of[0])

        sp_x[sp_x < 0] = 0
        sp_y[sp_y < 0] = 0
        sp_x[sp_x >= height] = height - 1
        sp_y[sp_y >= width] = width - 1

        shifted_labels[sp_x, sp_y] = sp_id
    return shifted_labels


def calculateVelocityAndOrientationVectors(labels, shifted_labels, avg_flow, 
                                               avg_fst_depth, avg_shifted_depth):
    """
    Calculate the velocity and the oriention of each superpixel 
    :param labels: original superpixel labels
    :param shifted_labels: shifted superpixel labels
    :param avg_flow: average optical flow matrix
    :param avg_fst_depth: average depth matrix of the original image
    :param avg_shifted_depth: average depth matrix of the shifted image
    :return: velocity (in km/h) and oriention matrix
    """
    velocity = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.float32)
    orientation = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.float32)
    for sp_id in np.unique(labels):
        sp_x, sp_y = np.unravel_index(np.argmax(labels == sp_id, axis=None), labels.shape)
        v, u = avg_flow[sp_x, sp_y]

        h_angle, v_angle = calcAngleOfView(width_to_focal[labels.shape[1]], u, v)
        depth = avg_fst_depth[sp_x, sp_y]
        x, y = calcSize(h_angle, v_angle, depth)
        x, y = (x * fps) * 3.6, (y * fps) * 3.6

        shifted_sp_x, shifted_sp_y = np.unravel_index(np.argmax(shifted_labels == sp_id, 
                                                                axis=None), labels.shape)
        shifted_depth = avg_shifted_depth[shifted_sp_x, shifted_sp_y]
        z = (shifted_depth - depth) * fps * 3.6

        velocity[labels == sp_id] = (x, y, z)
        orientation[labels == sp_id] = normalize((x, y, z))
    return velocity, orientation


def saveAsImage(out_fn, data, min_val=None, max_val=None):
    """
    Save the given matrix with the specified name using plasma colormap
    :param out_fn: name of the output image
    :param data: data matrix
    :param min_val: minimum value
    :param max_val: maximum value
    """
    if min_val is None and max_val is not None:
        min_val = -max_val
    plt.imsave(out_fn, data, cmap='plasma', vmin=min_val, vmax=max_val)


def read_velocity(vel_fn):
    """
    Read the given velocity file
    :param vel_fn: name of the velocity file
    :return: velocity values (in km/h) in matrix
    """
    velocity = np.load(vel_fn)
    velocity[velocity > max_velocity] = max_velocity
    velocity[velocity < -max_velocity] = -max_velocity
    return velocity


def compare(gt, data, labels):
    """
    Compare a ground truth and a predicted 3 dimensional matrix  
    :param gt: ground truth matrix
    :param data: predicted matrix
    :param labels: superpixel labels
    :return: error value of each superpixel, average error rate and maximum achievable error
    """
    assert gt.shape == data.shape
    
    n_channels = data.shape[2]
    masked_labels = np.copy(labels)
    for idx in range(n_channels):
        masked_labels[gt[:, :, idx] == 0] = 99999

    max_error = math.sqrt(n_channels) * 2 * max_velocity
    error = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.float)
    error_rate, count = 0., 0
    for sp_id in np.unique(masked_labels):
        if sp_id == 99999:
            continue
        sp_x, sp_y = np.unravel_index(np.argmax(masked_labels == sp_id, axis=None), 
                                      masked_labels.shape)
        l = np.linalg.norm(gt[sp_x, sp_y, :] - data[sp_x, sp_y, :], ord=2)
        error[labels == sp_id] = l
        error_rate += l / max_error
        count += 1
    error_rate /= count
    return error, error_rate, max_error
