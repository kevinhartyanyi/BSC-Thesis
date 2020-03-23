import os
import cv2
import numpy as np
import skvideo.io
import tqdm
import tables
from scipy import ndimage
import pandas
import speed.readFlowFile as readFlowFile
import math
import matplotlib.pyplot as plt
from PIL import Image
import speed.computeColor as computeColor
import imageio
from natsort import natsorted

TMP_IMG = '/home/kevin/workspace/pipeline_zero/tmp_img/'

RESULTS = 'results'
OTHER_DIR = os.path.join(RESULTS, 'other')
VL_DIR = os.path.join(RESULTS, 'velocity')
NP_DIR = os.path.join(RESULTS, 'numbers')
MASK_DIR = os.path.join(RESULTS, 'mask')
DRAW_DIR = os.path.join(RESULTS, 'draw')
SUPER_PIXEL_DIR = os.path.join(RESULTS, 'super_pixel')


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

def getResultDirs():
    results = {"Velocity": VL_DIR, "Mask": MASK_DIR, "Draw": DRAW_DIR, "SuperPixel": SUPER_PIXEL_DIR}
    return results

def calculate_shifted_labels(labels, avg_flow):
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


def calc_angle_of_view(focal_length, width, height):
    """
    Calculate the horizontal and veritcal angle of view for a given region
    :param focal_length: focal length of the image
    :param width: width of the region
    :param height: height of the region
    :return: horizontal and veritcal angle of view
    """
    horizontal_angle = 2 * np.degrees(np.tan((width / 2) / focal_length))
    vertical_angle = 2 * np.degrees(np.tan((height / 2) / focal_length))
    return horizontal_angle, vertical_angle

def calc_size(h_angle, v_angle, depth):
    """
    Calculate the size of a given region
    :param h_angle: horizontal angle of view
    :param v_angle: veritcal angle of view
    :param depth: distance in meters
    :return: width and height in meters
    """
    h_angle = np.radians(h_angle / 2)
    width = np.arctan(h_angle) * depth * 2

    v_angle = np.radians(v_angle / 2)
    height = np.arctan(v_angle) * depth * 2

    return width, height

def calculate_velocity_and_orientation_vectors(labels, shifted_labels, avg_flow, 
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

        h_angle, v_angle = calc_angle_of_view(width_to_focal[labels.shape[1]], u, v)
        depth = avg_fst_depth[sp_x, sp_y]
        x, y = calc_size(h_angle, v_angle, depth)
        x, y = (x * fps) * 3.6, (y * fps) * 3.6

        shifted_sp_x, shifted_sp_y = np.unravel_index(np.argmax(shifted_labels == sp_id, 
                                                                axis=None), labels.shape)
        shifted_depth = avg_shifted_depth[shifted_sp_x, shifted_sp_y]
        z = (shifted_depth - depth) * fps * 3.6

        velocity[labels == sp_id] = (x, y, z)
        orientation[labels == sp_id] = normalize((x, y, z))
    return velocity, orientation

def calculate_velocity_and_orientation_vectors_vectorised(of_mask, next_position, prev_position, flow, 
                                            fst_depth, snd_depth):
    """
    Calculate the velocity and the oriention of each superpixel 
    :param labels: original superpixel labels
    :param shifted_labels: shifted superpixel labels
    :param flow: average optical flow matrix
    :param fst_depth: average depth matrix of the original image
    :param snd_depth: average depth matrix of the shifted image
    :return: velocity (in km/h) and oriention matrix
    """
    velocity = np.zeros((fst_depth.shape[0], fst_depth.shape[1], 3), dtype=np.float32)
    orientation = np.zeros((fst_depth.shape[0], fst_depth.shape[1], 3), dtype=np.float32)
    h,w = fst_depth.shape

    fst_mono = fst_depth
    fst_mono[of_mask] = 0

    snd_mono_2 = np.full_like(fst_mono, 0)
    snd_mono_2 = snd_depth[prev_position[..., 0], prev_position[..., 1]]
    snd_mono_2[of_mask] = 0

    z = (snd_mono_2 - fst_mono) * fps * 3.6

    """
    TC = None
    h, w, _ = flow.shape
    of_hor = flow.copy()
    ofhor_left = of_hor[:, :int(w/2), 0]
    ofhor_right = of_hor[:, int(w/2):, 0]
    if np.mean(ofhor_left) * np.mean(ofhor_right) > 1: # left and right horizontal OF have the same sign, i.e. turning
        print("Turning")
        #of_left = np.abs(of_hor[:, :int(w/2),0])
        #of_right = np.abs(of_hor[:, int(w/2):,0])
        disp_left = md[i, :, :int(w/2)]
        disp_right = md[i, :, int(w/2):]       
        
        TC = abs(np.mean(ofhor_left)-np.mean(ofhor_right)) / np.mean(avg_md)
    """
    v = flow[:,:,0]
    u = flow[:,:,1]
    h_angle, v_angle = calc_angle_of_view(width_to_focal[fst_depth.shape[1]], u, v)
    depth = fst_depth
    x, y = calc_size(h_angle, v_angle, depth)
    x, y = (x * fps) * 3.6, (y * fps) * 3.6

    velocity[:,:,0] = y
    velocity[:,:,1] = x
    velocity[:,:,2] = z
    return velocity#, orientation

def calculate_velocity_and_orientation_vectors_vectorised_gt(flow, fst_depth, snd_depth):
    """
    Calculate the velocity and the oriention of each superpixel 
    :param labels: original superpixel labels
    :param shifted_labels: shifted superpixel labels
    :param flow: average optical flow matrix
    :param fst_depth: average depth matrix of the original image
    :param snd_depth: average depth matrix of the shifted image
    :return: velocity (in km/h) and oriention matrix
    """
    velocity = np.zeros((fst_depth.shape[0], fst_depth.shape[1], 3), dtype=np.float32)
    orientation = np.zeros((fst_depth.shape[0], fst_depth.shape[1], 3), dtype=np.float32)
    h,w = fst_depth.shape

    z = (snd_depth - fst_depth) * fps * 3.6

    v = flow[:,:,0]
    u = flow[:,:,1]
    h_angle, v_angle = calc_angle_of_view(width_to_focal[fst_depth.shape[1]], u, v)
    depth = fst_depth
    x, y = calc_size(h_angle, v_angle, depth)
    x, y = (x * fps) * 3.6, (y * fps) * 3.6

    velocity[:,:,0] = x
    velocity[:,:,1] = y
    velocity[:,:,2] = z
    return velocity#, orientation

# TODO np.meshgrid
def calc_bidi_errormap(flowAB, flowBA, tau=1.0, consistent_flow=False):
    """Calculates the inconsistent Optical Flow, and the transformation matrix given by the forward OF.
    
    Parameters
    ----------
    flowAB : np.ndarray
        Forward Optical Flow matrix
    flowBA : np.ndarray
        Backward Optical Flow matrix
    tau : float, optional
        Maximum allowed Eucledian distance between the original and the OF transformed pixels, by default 1.0
    consistent_flow : bool, optional
        If true then masks the transformation matrix with the inconsistency, by default False
    
    Returns
    -------
    (np.ndarray, np.ndarray)
        Binary mask of the inconsistency, and the transformation matrix given by the forward OF.

    Notes
    -----
    Source: https://github.com/DediGadot/PatchBatch/blob/master/pb_utils.py
    """
    h,w = flowAB.shape[0:2]
    x_mat = (np.expand_dims(range(w),0) * np.ones((h,1),dtype=np.int32)).astype(np.int32)
    y_mat = (np.ones((1,w),dtype=np.int32) * np.expand_dims(range(h),1)).astype(np.int32)

    d1 = flowAB
    r_cords = (y_mat + d1[:,:,1]).astype(np.int32)
    c_cords = (x_mat + d1[:,:,0]).astype(np.int32)
    r_cords[r_cords>h-1] = h-1
    r_cords[r_cords<0] = 0
    c_cords[c_cords>w-1] = w-1
    c_cords[c_cords<0] = 0
    next_positions = np.concatenate([r_cords[:, :, None], c_cords[:, :, None]], axis=-1)

    d2 = flowBA[r_cords,c_cords,:]
    d = np.sqrt(np.sum((d1+d2)**2,axis=2))

    bidi_map = d > tau

    if consistent_flow:
        next_positions[bidi_map, 0] = y_mat[bidi_map] 
        next_positions[bidi_map, 1] = x_mat[bidi_map] 

    d1 = flowBA
    r_cords = (y_mat - d1[:,:,1]).astype(np.int32)
    c_cords = (x_mat - d1[:,:,0]).astype(np.int32)
    r_cords[r_cords>h-1] = h-1
    r_cords[r_cords<0] = 0
    c_cords[c_cords>w-1] = w-1
    c_cords[c_cords<0] = 0
    prev_positions = np.concatenate([r_cords[:, :, None], c_cords[:, :, None]], axis=-1)

    return bidi_map, next_positions , prev_positions


def draw_velocity_vectors(img, disparities, max_spin_size=5.0, step_size=20, relative_disp=True, color=(0, 255, 0)):
    """
    Draw velocity vectors on the given image
    :param labels: superpixel labels
    :param velocity: velocity matrix
    :param img_fn: name of the image file
    :param draw_boundaries: whether to draw superpixel boundaries or not
    :return: image with velocity vectors
    """
    assert disparities.ndim == 3 and img.ndim == 3
    assert disparities.shape[:2] == img.shape[:2]
    assert disparities.shape[2] == 2 and img.shape[2] == 3
    
    height, width, _ = disparities.shape
    img = np.copy(img)  # height x width x channels
    disparities = disparities.astype(np.int32)
    if not relative_disp:
        x_mat = (np.expand_dims(range(width), 0) * np.ones((height, 1), dtype=np.int32)).astype(np.int32)
        y_mat = (np.ones((1, width), dtype=np.int32) * np.expand_dims(range(height), 1)).astype(np.int32)
        disparities[..., 0] -= y_mat
        disparities[..., 1] -= x_mat

    # Compute the maximum l (longest flow)
    l_max = 0.
    for x in range(0, height, step_size):
        for y in range(0, width, step_size):
            cx, cy = x, y
            dx, dy = disparities[cx, cy]
            l = math.sqrt(dx*dx + dy*dy)
            if l > l_max: 
                l_max = l

    # Draw arrows
    for x in range(0, height, step_size):
        for y in range(0, width, step_size):
            cx, cy = x, y
            dx, dy = disparities[cx, cy]
            l = math.sqrt(dx*dx + dy*dy)
            
            if l > 0:
                # Factor to normalise the size of the spin depeding on the length of the arrow
                spin_size = max_spin_size * l / l_max
                nx, ny = int(cx + dx), int(cy + dy)

                # print((nx, ny))
                nx = min(max(0, nx), height - 1) 
                ny = min(max(0, ny), width - 1)

                cx, cy = cy, cx
                nx, ny = ny, nx

                cv2.line(img, (cx, cy), (nx, ny), color, 1, cv2.LINE_AA)

                # Draws the spin of the arrow
                angle = math.atan2(cy - ny, cx - nx)

                cx = int(nx + spin_size * math.cos(angle + math.pi / 4))
                cy = int(ny + spin_size * math.sin(angle + math.pi / 4))
                cv2.line(img, (cx, cy), (nx, ny), color, 1, cv2.LINE_AA, 0)
                
                cx = int(nx + spin_size * math.cos(angle - math.pi / 4))
                cy = int(ny + spin_size * math.sin(angle - math.pi / 4))
                cv2.line(img, (cx, cy), (nx, ny), color, 1, cv2.LINE_AA, 0)
    return img

def save_as_image(out_fn, data, min_val=None, max_val=None):
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

def read_depth(depth_fn, width, height):
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
    #depth[depth > max_depth] = max_depth
    #depth[depth < min_depth] = min_depth
    return depth

def list_directory(dir_name, extension=None):
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
    fns = natsorted(fns)
    #fns.sort()
    return fns

def print_vectors(x,y,z):
    print("X mean: " + str(np.mean(x)))
    print("Y mean: " + str(np.mean(y)))
    print("Z mean: " + str(np.mean(z)))

def print_vectorsOF(x,y):
    print("X mean: " + str(np.mean(x)))
    print("Y mean: " + str(np.mean(y)))

def vector_distance(x,y,z):
    x2 = np.power(x, 2)
    y2 = np.power(y, 2)
    z2 = np.power(z, 2)
    added = x2 + y2 + z2
    return np.sqrt(added)

def vector_distanceOF(x,y):
    x2 = np.power(x, 2)
    y2 = np.power(y, 2)
    added = x2 + y2
    return np.sqrt(added)

def reduce_sort(vector, low=0.1,high=0.9, skip=None):
    v_unique = np.sort(vector.flatten())
    if skip is not None:
        v_unique = v_unique[v_unique != skip]

    mask_vector = np.logical_and(vector >= v_unique[int(len(v_unique) * low)],
        vector <= v_unique[int(len(v_unique) * high) - 1])
    return mask_vector


def vector_speed(vectors, low, high, slc = 0):
    x = vectors[:,:,0]
    y = vectors[:,:,1]
    z = vectors[:,:,2]
    #######################x
    #mask_z_thr = (z != 0)
    #mask_z_thr = (z < 0)

    #disp_mask = np.logical_and(disp >= disp_unique[int(len(disp_unique) * md_low)],
    #    disp <= disp_unique[int(len(disp_unique) * md_high) - 1])

    mask_y_thr = reduce_sort(y, low=low,high=high)
    #mask_z_thr = reduce_sort(z,low=0.3,high=0.6) # 0.5 - 0.6: 26
    only_good_of = z != 0
    z_negative = z < 0

    #print(np.unique(z[z_negative]))

    mask_uni = only_good_of
    #mask_uni = np.ones(x.shape, dtype=bool)
    #mask_uni = np.logical_and(mask_z_thr, only_good_of)
    mask_uni = np.logical_and(mask_y_thr, only_good_of)


    #######################x
    width = x.shape[1]
    half = int(width / 2)
    
    # Left Side
    mask_uni_left = mask_uni[:, 0:half]
    x_thr_left    = x       [:, 0:half] [mask_uni_left]
    y_thr_left    = y       [:, 0:half] [mask_uni_left]
    z_thr_left    = z       [:, 0:half] [mask_uni_left]

    print("Left Calculated:")
    print_vectors(x_thr_left,y_thr_left,z_thr_left)
    speed_left = vector_distance(x_thr_left,y_thr_left,z_thr_left)  
    print("Speed Left Calculated: " + str(np.mean(speed_left)))
    print("")

    # Left Side
    mask_uni_right = mask_uni[:, half:width]  
    x_thr_right    = x       [:, half:width] [mask_uni_right]
    y_thr_right    = y       [:, half:width] [mask_uni_right]
    z_thr_right    = z       [:, half:width] [mask_uni_right]

    print("Right Calculated:")
    print_vectors(x_thr_right,y_thr_right,z_thr_right)
    speed_right = vector_distance(x_thr_right,y_thr_right,z_thr_right)  
    print("Speed Right Calculated: " + str(np.mean(speed_right)))
    print("")

    print("Average:")
    speed_avg = np.average([np.mean(speed_left), np.mean(speed_right)])  
    print("Speed Average: " + str(np.mean(speed_avg)))
    print("")

    return speed_avg, mask_uni

    """
    if slc == 0: 
        print("Calculated:")
        print_vectors(x_thr,y_thr,z_thr)
        speed = vector_distance(x_thr,y_thr,z_thr)  
        print("Speed Calculated: " + str(np.mean(speed)))
        print("")
        speed_img = vector_distance(x,y,z)  
        return speed_img, mask_uni"""
    """
    if slc == 2: 
        x = vectors[:,:,0]
        height = x.shape[0]
        height_2 = int(height / 2)
        mask_z_thr = mask_z_thr[height_2:height,:]
        print(mask_z_thr)
        x = vectors[height_2:height,:,0][mask_z_thr]
        y = vectors[height_2:height,:,1][mask_z_thr]
        z = vectors[height_2:height,:,2][mask_z_thr]
        print("Lower:")
        print_vectors(x,y,z)
        x = np.power(x, 2)
        y = np.power(y, 2)
        z = np.power(z, 2)
        add2 = x + y + z
        mLower = np.sqrt(add2)    
        speed2 = np.mean(mLower)    
        print("Speed Lower: " + str(speed2))
        print("")
    if slc == 1: 
        x = vectors[:,:,0]
        height = x.shape[0]        
        mask_z_thr = mask_z_thr[0:height,:]
        x = vectors[0:height,:,0][mask_z_thr]
        y = vectors[0:height,:,1][mask_z_thr]
        z = vectors[0:height,:,2][mask_z_thr]
        print("Upper:")
        print_vectors(x,y,z)
        x = np.power(x, 2)
        y = np.power(y, 2)
        z = np.power(z, 2)
        add3 = x + y + z
        mUpper = np.sqrt(add3)    
        speed3 = np.mean(mUpper)    
        print("Speed Upper: " + str(speed3))
        print("")"""
    """x = vectors[:,:,0]
    y = vectors[:,:,1]
    z = vectors[:,:,2]
    if slc == 0: 
        x = np.power(x, 2)
        y = np.power(y, 2)
        z = np.power(z, 2)
        add1 = x + y + z
        mOriginal = np.sqrt(add1)
        speed1 = np.mean(mOriginal)    
    if slc == 2: 
        height = x.shape[0]
        height_2 = int(height / 2)
        x = vectors[height_2:height,:,0]
        y = vectors[height_2:height,:,1]
        z = vectors[height_2:height,:,2]
        x = np.power(x, 2)
        y = np.power(y, 2)
        z = np.power(z, 2)
        add2 = x + y + z
        mLower = np.sqrt(add2)    
        speed2 = np.mean(mLower)    
    if slc == 1: 
        height = x.shape[0]
        x = vectors[0:height,:,0]
        y = vectors[0:height,:,1]
        z = vectors[0:height,:,2]
        x = np.power(x, 2)
        y = np.power(y, 2)
        z = np.power(z, 2)
        add3 = x + y + z
        mUpper = np.sqrt(add3)    
        speed3 = np.mean(mUpper)"""    

def vector_speedOF(vectors, slc = 0):
    x = vectors[:,:,0]
    y = vectors[:,:,1]
    #######################x

    #disp_mask = np.logical_and(disp >= disp_unique[int(len(disp_unique) * md_low)],
    #    disp <= disp_unique[int(len(disp_unique) * md_high) - 1])

    mask_y_thr = reduce_sort(y)
    only_good_of = x != 0
    x_negative = x < 0

    #print(np.unique(z[z_negative]))


    mask_uni = only_good_of    
    mask_uni = np.ones(x.shape, dtype=bool)
    #mask_uni = np.logical_and(mask_z_thr, only_good_of)


    #######################x
    width = x.shape[1]
    half = int(width / 2)
    
    # Left Side
    mask_uni_left = mask_uni[:, 0:half]
    x_thr_left    = x       [:, 0:half] [mask_uni_left]
    y_thr_left    = y       [:, 0:half] [mask_uni_left]

    print("Left Calculated:")
    print_vectorsOF(x_thr_left,y_thr_left)
    speed_left = vector_distanceOF(x_thr_left,y_thr_left)  
    print("Speed Left Calculated: " + str(np.mean(speed_left)))
    print("")

    # Left Side
    mask_uni_right = mask_uni[:, half:width]  
    x_thr_right    = x       [:, half:width] [mask_uni_right]
    y_thr_right    = y       [:, half:width] [mask_uni_right]

    print("Right Calculated:")
    print_vectorsOF(x_thr_right,y_thr_right)
    speed_right = vector_distanceOF(x_thr_right,y_thr_right)   
    print("Speed Right Calculated: " + str(np.mean(speed_right)))
    print("")

    print("Average:")
    speed_avg = np.average([np.mean(speed_left), np.mean(speed_right)])  
    print("Speed Average: " + str(np.mean(speed_avg)))
    print("")


    speed_img = vector_distanceOF(x,y)  
    return speed_img, mask_uni


def vector_speedOF4Side(vectors, slc = 0):
    x = vectors[:,:,0]
    y = vectors[:,:,1]
    #######################x

    #disp_mask = np.logical_and(disp >= disp_unique[int(len(disp_unique) * md_low)],
    #    disp <= disp_unique[int(len(disp_unique) * md_high) - 1])

    mask_y_thr = reduce_sort(y)
    only_good_of = x != 0
    x_negative = x < 0

    #print(np.unique(z[z_negative]))


    mask_uni = only_good_of    
    mask_uni = np.ones(x.shape, dtype=bool)
    #mask_uni = np.logical_and(mask_z_thr, only_good_of)


    #######################x
    width = x.shape[1]
    height = x.shape[0]
    half = int(width / 2)
    half_height =  int(height / 2)
    
    # Left Side
    mask_uni_left = mask_uni[:, 0:half]
    x_thr_left    = x       [:, 0:half] [mask_uni_left]
    y_thr_left    = y       [:, 0:half] [mask_uni_left]

    print("Left Calculated:")
    print_vectorsOF(x_thr_left, y_thr_left)
    speed_left = abs(x_thr_left)
    print("Speed Left Calculated: " + str(np.mean(speed_left)))
    print("")

    # Right Side
    mask_uni_right = mask_uni[:, half:width]  
    x_thr_right    = x       [:, half:width] [mask_uni_right]
    y_thr_right    = y       [:, half:width] [mask_uni_right]

    print("Right Calculated:")
    print_vectorsOF(x_thr_right, y_thr_left)
    speed_right = abs(x_thr_right) 
    print("Speed Right Calculated: " + str(np.mean(speed_right)))
    print("")

    # Up Side
    mask_uni_up    = mask_uni[0:half_height,:]  
    y_thr_up       = y       [0:half_height,:] [mask_uni_up]
    x_thr_up       = x       [0:half_height,:] [mask_uni_up]

    print("Up Calculated:")
    print_vectorsOF(x_thr_up, y_thr_up)
    speed_up = abs(y_thr_up) 
    print("Speed Up Calculated: " + str(np.mean(speed_up)))
    print("")

    # Down Side
    mask_uni_down    = mask_uni[half_height:height,:]  
    y_thr_down       = y       [half_height:height,:] [mask_uni_down]
    x_thr_down       = x       [half_height:height,:] [mask_uni_down]

    print("Down Calculated:")
    print_vectorsOF(x_thr_down, y_thr_down)
    speed_down = abs(y_thr_down) 
    print("Speed Down Calculated: " + str(np.mean(speed_down)))
    print("")

    print("Average:")
    speed_avg = np.average([np.mean(speed_left), np.mean(speed_right), np.mean(speed_up), np.mean(speed_down)])  
    print("Speed Average: " + str(np.mean(speed_avg)))
    print("")

    print("Average v2:")
    speed_avg = np.average([np.mean(np.abs(x)), np.mean(np.abs(y))])  
    print("Speed Average: " + str(np.mean(speed_avg)))
    print("")

    speed_img = vector_distanceOF(x,y)  
    return speed_img, mask_uni

def vector_speedOF_Simple(vectors, high=1, low=0):
    x = vectors[:,:,0]
    y = vectors[:,:,1]

    #mask_y_thr = reduce_sort(y)
    only_good_of = x != 0
    #x_negative = x < 0

    mask_uni = only_good_of    
    mask_y_thr = reduce_sort(y,low=low,high=high) # 0.5 - 0.6: 26

    # Speed throwing
    """
    x_0 = x.copy()
    y_0 = y.copy()
    x_0[~only_good_of] = 1000
    y_0[~only_good_of] = 1000
    x_abs_0 = np.abs(x_0)
    y_abs_0 = np.abs(y_0)
    avg_speed_raw = np.divide(np.add(x_abs_0, y_abs_0), 2)
    mask_speed_thr = reduce_sort(avg_speed_raw,low=low,high=high, skip=1000)
    #save_as_image('speed_mask_test.png', mask_speed_thr, min_val=0, max_val=max_depth) 
    mask_uni = np.logical_and(mask_speed_thr, only_good_of)
    """

    mask_uni = np.logical_and(mask_y_thr, only_good_of)


    x_thr = x[mask_uni]
    y_thr = y[mask_uni]


    x_abs = np.abs(x_thr)
    y_abs = np.abs(y_thr)


    #######################x Turning Compensation
    TC = None
    h, w = x.shape
    of_hor = x.copy()
    ofhor_left = of_hor[:, :int(w/2)]
    ofhor_right = of_hor[:, int(w/2):]
    
    if np.mean(ofhor_left) * np.mean(ofhor_right) > 1: # left and right horizontal OF have the same sign, i.e. turning
        maske_uni_left = mask_uni[:, :int(w/2)]
        maske_uni_right = mask_uni[:, int(w/2):]
        md = vectors[:,:,2]   
        ofhor_left_thr = ofhor_left[maske_uni_left]
        ofhor_right_thr = ofhor_right[maske_uni_right]
        md_thr = md[mask_uni]
        
        TC = np.mean([abs(np.mean(ofhor_left)-np.mean(ofhor_right)), np.mean(y_abs)])
        #TC = np.mean([abs(np.mean(ofhor_left_thr)-np.mean(ofhor_right_thr)), np.mean(y_abs)])

        #print("Turning")
        return TC, mask_uni
    
    
    #print("Not Turning")
    #######################x

    #print("Average:")
    speed_avg = np.average([np.mean(x_abs), np.mean(y_abs)])   
    speed = np.mean(speed_avg) # Don't ask why this is here

    #print("Raw ", np.mean(avg_speed_raw[avg_speed_raw != 1000]))
    #print("Speed ", speed)s

    return speed, mask_uni

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
    #assert n_sps == -1 or n_sps == len(np.unique(labels))
    return labels

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

def concat_vid2(vid1_path, vid2_path, out_path, dir = 0, test = False, x = 0, y = 0, w = 0, h = 0):
    '''
    Concatenates two videos side by side; videos must be same resolution.
    Input parameters:
        vid1_path = full path of input video 1
        vid2_path = full path of input video 2
        out_path = full path of output video
        dir = direction of concatenation: 0=vertical, 1=horizontal(side-by-side)
        test = for debugging purposes, if True result video is not saved
        x, y, w, h = determine a rectangle, for debugging purposes
    '''
    if not (os.path.exists(vid1_path) and os.path.exists(vid2_path)):
        raise ValueError('Input video file does not exist.')
    cap1 = cv2.VideoCapture(vid1_path)
    cap2 = cv2.VideoCapture(vid2_path)
    frameNr = int(cap1.get(7))
    fps = cap1.get(5)
    fourcc = int(cap1.get(cv2.CAP_PROP_FOURCC))
    width1, height1 = int(cap1.get(3)), int(cap1.get(4))
    width2, height2 = int(cap2.get(3)), int(cap2.get(4))
    if dir == 0:
        assert width1 == width2
        width = width1
        height = height1 + height2
    elif dir == 1:
        assert height1 == height2
        width = width1 + width2
        height = height1
    else:
        raise ValueError('Unknown concatenation direction!')
    
    if test:
        cv2.namedWindow('im', cv2.WINDOW_NORMAL)
        cv2.moveWindow('im', 150, 20)
        cv2.resizeWindow('im', width, height)
    else:
        out = skvideo.io.FFmpegWriter(out_path)

    with tqdm.tqdm(total=frameNr) as pbar:
        for i in range(frameNr):
            ret1, f1 = cap1.read()
            ret2, f2 = cap2.read()
            if not ret1 or not ret2:
            	break
            if test:
            	cv2.rectangle(f1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            	cv2.rectangle(f2, (x, y), (x+w, y+h), (0, 255, 0), 2)
            im_big = np.concatenate((f1, f2), axis = dir)
            if test:
            	cv2.imshow('im', im_big)
            	if cv2.waitKey() == 27:
            		break
            else:
            	out.writeFrame(cv2.cvtColor(im_big, cv2.COLOR_BGR2RGB))
            pbar.update()
        
    cap1.release()
    cap2.release()
    if not test:
        out.close()

def extract_rect_of(hdf5_in, hdf5_out, x=0, y=0, w=200, h=120):
    '''
    Extracts middle of Optical Flow values determined by a rectangle.
    Output is a new .hdf5 file with a smaller resolution corresponding to the rectangle size.
    Input parameters:
        hdf5_path = full path of input .hdf5 OF annotations
        out_path = full path of output .hdf5 OF annotations
        x, y = coordinates of the upper left corner of the rectangle
        w, h = determine the width and height of the rectangle to extract   
    '''
    hdf5_file_in = tables.open_file(hdf5_in, mode='r')
    flows = hdf5_file_in.root.flows[:]
    N = flows.shape[3]
    
    hdf5_file_out = tables.open_file(hdf5_out, mode='w')
    flows_out = hdf5_file_out.create_earray(hdf5_file_out.root, 'flows', 
                                    atom = tables.Atom.from_dtype(np.dtype('Float32')), 
                                    shape = (h, w, 2, 0), 
                                    filters = tables.Filters(complevel=5, complib='blosc'), 
                                    expectedrows = N)
    for i in range(N):
        flows_out.append(np.expand_dims(flows[y:y+h, x:x+w, :, i], axis=3))
            
    hdf5_file_in.close()
    hdf5_file_out.close()

def extract_rect_depth(npy_in, npy_out, x=0, y=0, w=200, h=120):
    '''
    Extracts middle of Depth values determined by a rectangle.
    Output is a new .npy file with a smaller resolution corresponding to the rectangle size.
    Input parameters:
        npy_in = .npy file containing input Depth results
        npy_in = .npy file containing output Depth results
        x, y = coordinates of the upper left corner of the rectangle
        w, h = determine the width and height of the rectangle to extract   
    '''
    md = np.load(npy_in)
    np.save(npy_out, md[:, y:y+h, x:x+w])

def extract_part_rectangle(vid_path, out_path, test = False, x = 0, y = 0, w = 160, h = 90):
    '''
    Extracts middle of frames from a video determined by a rectangle.
    Output is a new video with a smaller resolution corresponding to the rectangle size.
    Input parameters:
        vid_path = full path of input video
        out_path = full path of output video
        test = for debugging purposes to see the rectangle position on the original frames, if True result video is not saved
        x, y = coordinates of the upper left corner of the rectangle
        w, h = determine the width and height of the rectangle to extract
    '''
    if not os.path.exists(vid_path):
        raise ValueError('Input video file does not exist.')
    cap = cv2.VideoCapture(vid_path)
    frameNr = int(cap.get(7))
    fps = cap.get(5)

    if test:
        cv2.namedWindow('im', cv2.WINDOW_NORMAL)
        cv2.moveWindow('im', 150, 20)
    else:
        vid_part = skvideo.io.FFmpegWriter(out_path)

    with tqdm.tqdm(total=frameNr) as pbar:
        for i in range(frameNr):
            ret, frame = cap.read()
            if not ret:
                break
            if test:
                cv2.resizeWindow('im', frame.shape[1], frame.shape[0])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow('im', frame)
                if cv2.waitKey() == 27:
                    break
            else:
                vid_part.writeFrame(cv2.cvtColor(frame[y:y+h, x:x+w, :], cv2.COLOR_BGR2RGB))
            pbar.update()
    cap.release()
    if test:
        cv2.destroyAllWindows()
    else:
        vid_part.close()

def crop(vid_path, pwcnet_path, depth_path, out_path, crop=[700,100,400,240]):
    """[Crop video and save new video, new of and new depth]
    
    Args:
        vid_path ([str]): [path to video]
        pwcnet_path ([str]): [path to pwcnet (hdf5)]
        depth_path ([str]): [path to monodepth (npy)]
        crop (list, optional): [coordinates of the cropped video.]. Defaults to [700,100,400,240].
    """
    x, y, w, h = crop
    print('Extracting crop from KITTI video...')
    extract_part_rectangle(vid_path, out_path + "cropped.mp4", False, x, y, w, h)
    print('Extracting crop from flow results...')
    extract_rect_of(pwcnet_path, out_path + "pwcnet_crop.hdf5", x, y, w, h)
    print('Extracting crop from depth results...')
    extract_rect_depth(depth_path, out_path + "monodepth_crop.npy", x, y, w, h)


def error_comparison(speed_est, speed_gt, speed_est_boruvka=None, multiple_est=False, csv=None):
    """[Visualize error between estimated speed and the ground truth speed]
    
    Args:
        speed_est ([type]): [The estimated speed]
        speed_gt ([type]): [Ground truth speed]
        speed_est_boruvka ([type], optional): [If not None, then also compare speed_est and speed_est_boruvka]. Defaults to None.
        multiple_est (bool, optional): [If you have more than one value for every frame in speed_est, then set this to True ]. Defaults to False.
        csv ([str], optional): [If not None, then save results into a csv file with the path you entered.]. Defaults to None.
    """
    if multiple_est:
        speed_gt = np.repeat(np.expand_dims(speed_gt, 1), speed_est.shape[1], 1)

    scale = np.sum(np.multiply(speed_est, speed_gt), 0) / np.sum(np.multiply(speed_est, speed_est), 0) # Set multiple_est to True if you have more than one speed value
    rmse = np.sqrt(np.mean(np.square(speed_est*scale-speed_gt), 0)) # Root-mean-square deviation

    if multiple_est:
        rmse_sorted = np.sort(rmse)
        ind = np.argsort(rmse)
    
    csv_list = []

    if multiple_est:
        print("Scale:\n" + str (np.around(scale,3)))
        print("RMSE:\n" + str (np.around(rmse,3)))
        print("RMSE Sorted:\n" + str (np.around(rmse_sorted,3)))
        print("Indexes:\n" + str(ind))
    else:
        spaces = "        "
        headline = "Truth"
        if speed_est_boruvka is None:
            headline += spaces + "Est" + spaces + "Scaled" + spaces + "Error"
        else:
            headline += spaces + "Simple" + spaces + "Scaled" + spaces + "Error" + spaces + "Boruvka" + spaces + "Scaled" + spaces + "Error" + spaces + "Better"
        

        print(headline)
        if speed_est_boruvka is None:
            for frame in range(0, len(speed_gt)):
                print(("{:.2f}" + spaces + "{:.2f}" + spaces + "{:.2f}" + spaces + "{:.2f}")
                .format(speed_gt[frame], speed_est[frame], speed_est[frame]*scale, speed_gt[frame]-(speed_est[frame]*scale)))
                csv_list.append({"Truth":speed_gt[frame], "Est": speed_est[frame], "Scaled": speed_est[frame]*scale, "Error": speed_gt[frame]-(speed_est[frame]*scale)})
            print(("Scale: {:.3f}").format(scale))
            print(("RMSE: {:.3f}").format(rmse))
            csv_list.append({"Scale": scale, "RMSE": rmse})
        else:
            scale_boruvka = np.sum(np.multiply(speed_est_boruvka, speed_gt), 0) / np.sum(np.multiply(speed_est_boruvka, speed_est_boruvka), 0)
            rmse_boruvka = np.sqrt(np.mean(np.square(speed_est_boruvka*scale_boruvka-speed_gt), 0)) # Root-mean-square deviation
            for frame in range(0, len(speed_gt)):
                better = ""
                if abs(speed_gt[frame]-(speed_est[frame]*scale)) < abs(speed_gt[frame]-(speed_est_boruvka[frame]*scale_boruvka)):
                    better = "Simple"
                else:
                    better = "Boruvka"
                print(("{:.2f}" + spaces + "{:.2f}" + spaces + "{:.2f}" + spaces + "{:.2f}" + spaces + "{:.2f}" + spaces + "{:.2f}" 
                + spaces + "{:.2f}"  + spaces + "{}")
                .format(speed_gt[frame], speed_est[frame], speed_est[frame]*scale, speed_gt[frame]-(speed_est[frame]*scale),
                speed_est_boruvka[frame], speed_est_boruvka[frame]*scale_boruvka, speed_gt[frame]-(speed_est_boruvka[frame]*scale_boruvka), better))
                csv_list.append({"Truth":speed_gt[frame], "Simple": speed_est[frame], "Scaled": speed_est[frame]*scale, "Error": speed_gt[frame]-(speed_est[frame]*scale),
                "Boruvka": speed_est_boruvka[frame], "Scaled": speed_est_boruvka[frame]*scale_boruvka, "Error": speed_gt[frame]-(speed_est_boruvka[frame]*scale_boruvka),
                "Better": better})
            print(("Scale Simple: {:.3f}\n"  + "Scale Boruvka: {:.3f}").format(scale, scale_boruvka))
            print(("RMSE Simple: {:.3f}\n"  + "RMSE Boruvka: {:.3f}").format(rmse, rmse_boruvka))
            better = ""
            if rmse < rmse_boruvka:
                better = "Simple"
            else:
                better = "Boruvka"
            print("==> " + better)
            csv_list.append({"Scale Simple": scale, "Scale Boruvka": scale_boruvka, "RMSE Simple": rmse, "RMSE Boruvka": rmse_boruvka})
        if csv is not None:
            csv_file = pandas.DataFrame(csv_list)
            csv_file.index.name = "Frame"
            csv_file.to_csv(csv, header=True)


def scale_speed_values(speed_est, speed_gt, multiple_est=False):
    """[Scale the estimated speed (returned by speed_calculation_simple and speed_calculation_boruvka)
    to be comparable with the ground truth speed]
    
    Args:
        speed_est ([numpy array]): [The estimated speed]
        speed_gt ([numpy array]): [Ground truth speed]
        multiple_est (bool, optional): [If you have more than one value for every frame in speed_est, then set this to True ]. Defaults to False.
    
    Returns:
        [numpy array]: [The scaled speed]
    """
    # compute scale minimizing RMSE
    if multiple_est:
        speed_gt = np.repeat(np.expand_dims(speed_gt, 1), speed_est.shape[1], 1)

    scale = np.sum(np.multiply(speed_est, speed_gt), 0) / np.sum(np.multiply(speed_est, speed_est), 0) # Set multiple_est to True if you have more than one speed value
    return speed_est * scale


def error_comparison_Speed_Vecors(speed_est, speed_gt, csv=None, visualize=True):
    """[Visualize error between estimated speed and the ground truth speed]
    
    Args:
        speed_est ([type]): [The estimated speed]
        speed_gt ([type]): [Ground truth speed]
        csv ([str], optional): [If not None, then save results into a csv file with the path you entered.]. Defaults to None.
    """
    rmse = np.sqrt(np.mean(np.square(speed_est-speed_gt), 0)) # Root-mean-square deviation
    if visualize is False:
        return rmse
    csv_list = []
    
    spaces = "        "
    headline = "Truth"
    headline += spaces + "Est" + spaces + "Error"
    

    print(headline)
    for frame in range(0, len(speed_gt)):
        print(("{:.2f}" + spaces + "{:.2f}" + spaces + "{:.2f}")
        .format(speed_gt[frame], speed_est[frame], speed_gt[frame]-(speed_est[frame])))
        csv_list.append({"Truth":speed_gt[frame], "Est": speed_est[frame],"Error": speed_gt[frame]-(speed_est[frame])})

    print(("RMSE: {:.3f}").format(rmse))
    csv_list.append({"RMSE": rmse})
    
    if csv is not None:
        csv_file = pandas.DataFrame(csv_list)
        csv_file.index.name = "Frame"
        csv_file.to_csv(csv, header=True)

    return rmse

def create_video_of_or_mono(path, data, mono=False):
    if mono:                      
        #disp_fns = list_directory(data, extension='.npy')
        #read_depth(disp_fns, width, height)
        imgs = list_directory(data, extension='.png')
        depth_img = []
        for file in imgs:
            depth_img.append(np.asarray(Image.open(file)))
        imageio.mimwrite(path, np.asarray(depth_img) , fps=30)
    else:                
        flow_fns = list_directory(data, extension='.flo')
        flow = []
        for file in tqdm.tqdm(flow_fns):
            flow.append(computeColor.computeImg(readFlowFile.read(file)))
        imageio.mimwrite(path, np.asarray(flow) , fps=30)       

def concat_video3(vid_path, vid_path2, vid_path3, out_path):
    vid = cv2.VideoCapture(vid_path) #  375 x 1242
    vid_pwcnet = cv2.VideoCapture(vid_path2)       #  384 x 1248
    vid_hd3 = cv2.VideoCapture(vid_path3) #  374 x 1242

    frame_nr = int(vid.get(7))
    vid_out = skvideo.io.FFmpegWriter(out_path)
    
    print('Make Video')
    for i in tqdm.tqdm(range(frame_nr-1)):
        _, f = vid.read()
        _, f_flow = vid_pwcnet.read()
        _, f_hd3 = vid_hd3.read()
        
        z = np.zeros((1167, 1270, 3), dtype=np.uint8)
        z[15:390, 20:1262, :] = f
        z[390:774, 20:1268, :] = f_flow
        z[783:1157, 20:1262, :] = f_hd3
        vid_out.writeFrame(cv2.cvtColor(z, cv2.COLOR_BGR2RGB))
        #cv2.imshow('im', z)
        #if cv2.waitKey() == 27:
        #    break

def create_speed_video_Speed_Vectors(vid_path, out_path, speed_simple, speed_gt, mask, flow, back_flow, depth, hd3=False):
    """[Display speed estimations, and vizualization.]
    
    Args:
        vid_path ([str]): [path to the video]
        out_path ([str]): [path where to save the speed video]
        speed_simple ([numpy array]): [speed values of simple speed (get this from speed_calculation_simple)]
        speed_boruvka ([numpy array]): [speed values of boruvka speed (get this from speed_calculation_boruvka)]
        speed_tc ([numpy array]): [speed with turn compensation]
        speed_gt ([numpy array]): [ground truth speed of the video]
        mask ([numpy array]): [mask of the pixels used for speed estimation]
    """
    '''
    vid_id = '2011_09_26_drive_0059'
    
    vid_path = DB_PATH[:-5] + vid_id + '.mp4'
    pwcnet = DB_PATH + '/speed_400/' + vid_id + '/pwcnet.mp4'
    monodepth = DB_PATH + '/speed_400/' + vid_id + '/monodepth.mp4'
    crop = [700,100,400,240]
    '''
    
    vid = cv2.VideoCapture(vid_path)            # 375 x 1242
    vid_pwcnet = cv2.VideoCapture(flow)       #  384 x 1248
    vid_back_pwcnet = cv2.VideoCapture(back_flow) #  384 x 1248
    vid_monodepth = cv2.VideoCapture(depth) #  384 x 1248
    frame_nr = int(vid.get(7))
    vid_out = skvideo.io.FFmpegWriter(out_path)
    
    print('Make Video')
    for i in tqdm.tqdm(range(frame_nr-1)):
        _, f = vid.read()
        _, f_flow = vid_pwcnet.read()
        _, f_back_flow = vid_back_pwcnet.read()
        _, f_depth = vid_monodepth.read()
        #cv2.rectangle(f, (crop[0], crop[1]), (crop[0]+crop[2], crop[1]+crop[3]), (255, 255, 255), 2)
        frame_mask = mask[i] # 375 x 1242
        #save_as_image(TMP_IMG + str(i) + '.png', frame_mask)
        frame_mask = np.expand_dims(frame_mask, axis=-1)
        z = np.zeros((1167, 2548, 3), dtype=np.uint8)
        if hd3 is False:
            z[15:390, 20:1262, :] = f
            z[390:765, 20:1262, :] = frame_mask*100
            z[15:399, 1300:2548, :] = f_flow
            z[399:783, 1300:2548, :] = f_back_flow
            z[783:1167, 650:1898, :] = f_depth
        else:
            z[15:390, 20:1262, :] = f
            z[390:765, 20:1262, :] = frame_mask*100
            z[15:389, 1300:2542, :] = f_flow
            z[389:763, 1300:2542, :] = f_back_flow
            z[783:1167, 650:1898, :] = f_depth
        font = cv2.FONT_HERSHEY_DUPLEX
        fSize = 0.7
        fThick = 1
        #cv2.putText(z, 'FlowNet2', (450, 440), font, fSize, (255, 255, 255), fThick)
        #cv2.putText(z, 'MonoDepth', (900, 440), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, 'Frame ' + str(i), (50, 450), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, 'Speed estimation', (50, 480), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, 'GT:   ' + str(round(speed_gt[i], 3)), (50, 510), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, ' simple: ' + str(np.round(speed_simple[i], 3)), (50, 570), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, '    err: ' + str(round(speed_gt[i] - speed_simple[i], 3)), (220, 570), font, fSize, (255, 255, 255), fThick)
        #str(round(gazeDataGyro[iz, 4], 2))
        #vid_out.writeFrame(coloured)
        #vid_out.writeFrame(coloured)
        vid_out.writeFrame(cv2.cvtColor(z, cv2.COLOR_BGR2RGB))
        #cv2.imshow('im', z)
        #if cv2.waitKey() == 27:
        #    break

def create_speed_video(vid_path, out_path, speed_simple, speed_boruvka, speed_tc, speed_gt, mask):
    """[Display speed estimations, and vizualization.]
    
    Args:
        vid_path ([str]): [path to the video]
        out_path ([str]): [path where to save the speed video]
        speed_simple ([numpy array]): [speed values of simple speed (get this from speed_calculation_simple)]
        speed_boruvka ([numpy array]): [speed values of boruvka speed (get this from speed_calculation_boruvka)]
        speed_tc ([numpy array]): [speed with turn compensation]
        speed_gt ([numpy array]): [ground truth speed of the video]
        mask ([numpy array]): [mask of the pixels used for speed estimation]
    """
    '''
    vid_id = '2011_09_26_drive_0059'
    
    vid_path = DB_PATH[:-5] + vid_id + '.mp4'
    pwcnet = DB_PATH + '/speed_400/' + vid_id + '/pwcnet.mp4'
    monodepth = DB_PATH + '/speed_400/' + vid_id + '/monodepth.mp4'
    crop = [700,100,400,240]
    '''
    #import pickle
    #pickle_in = open('KITTI_speed.pkl', 'rb')
    #kitti_speed = pickle.load(pickle_in)
    #print(kitti_speed)
    #v_gt = kitti_speed['GT_' + vid_id]
    #v_est_crop = kitti_speed['speed_400_wo_' + vid_id]
    #v_est_full = kitti_speed['speed_1242_wo_' + vid_id]


    
    vid = cv2.VideoCapture(vid_path)            # 1242 x 375
    #vid_pwcnet = cv2.VideoCapture(pwcnet)       #  400 x 240
    #vid_monodepth = cv2.VideoCapture(monodepth) #  400 x 240
    frame_nr = int(vid.get(7))
    vid_out = skvideo.io.FFmpegWriter(out_path)
    
    for i in range(frame_nr):
        print('Handling frame: ', i)
        _, f = vid.read()
        '''_, f_flow = vid_pwcnet.read()
        _, f_depth = vid_monodepth.read()
        cv2.rectangle(f, (crop[0], crop[1]), (crop[0]+crop[2], crop[1]+crop[3]), (255, 255, 255), 2)'''

        frame_mask = mask[i]
        # create colormap and normalization
        cmap = plt.cm.spring
        norm = plt.Normalize(vmin=frame_mask[frame_mask != 0].min(), vmax=frame_mask[frame_mask != 0].max())
        # map the normalized data to colors
        coloured = cmap(norm(frame_mask))
        coloured = coloured[...,:3] * 255
        f[frame_mask != 0] = coloured[frame_mask != 0]
        z = np.zeros((720, 1280, 3), dtype=np.uint8)
        z[15:390, 20:1262, :] = f
        #z[450:690, 400:800, :] = f_flow
        #z[450:690, 850:1250, :] = f_depth
        font = cv2.FONT_HERSHEY_DUPLEX
        fSize = 0.7
        fThick = 1
        #cv2.putText(z, 'FlowNet2', (450, 440), font, fSize, (255, 255, 255), fThick)
        #cv2.putText(z, 'MonoDepth', (900, 440), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, 'Frame ' + str(i), (50, 450), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, 'Speed estimation', (50, 480), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, 'GT:   ' + str(round(speed_gt[i], 3)), (50, 510), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, ' boruvka: ' + str(round(speed_boruvka[i], 3)), (50, 540), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, '    err: ' + str(round(speed_gt[i] - speed_boruvka[i], 3)), (220, 540), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, ' simple: ' + str(round(speed_simple[i], 3)), (50, 570), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, '    err: ' + str(round(speed_gt[i] - speed_simple[i], 3)), (220, 570), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, ' TC: ' + str(round(speed_tc[i], 3)), (50, 600), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, '    err: ' + str(round(speed_gt[i] - speed_tc[i], 3)), (220, 600), font, fSize, (255, 255, 255), fThick)
        #str(round(gazeDataGyro[iz, 4], 2))
        #vid_out.writeFrame(coloured)
        #vid_out.writeFrame(coloured)
        vid_out.writeFrame(cv2.cvtColor(z, cv2.COLOR_BGR2RGB))
        #cv2.imshow('im', z)
        #if cv2.waitKey() == 27:
        #    break

    vid.release()
    #vid_pwcnet.release()
    #vid_monodepth.release()
    vid_out.close()

def gen_numbers(value, prev_step_size, gen=10):
    low = value - prev_step_size
    high = value + prev_step_size
    new_step_size = (high - low) / gen
    generated = [low + i*new_step_size for i in range(gen+1)]
    return generated, new_step_size



def get_speed_curves(hdf5_path, npy_path, filt_size, filt_mode = 'median', flow_thresh = 0.1, disp_thresh = 0.02):
    '''
    Calculate speed curves based on optical flow and monodepth 
    (division of means, mean of divisions with smoothing filter applied or outlier pixels removed)
    
    Arguments:
        hdf5_path:      path to HDF5 file containing OF results
        npy_path:       path to .npy file containing Monodepth results
        filt_size: temporal smoothing filter size
        filt_mode: filter mode: 'median', 'avg', 'gaussian' etc. (you name it)
        flow_thresh: threshold for optical flow to use, values greater than this will be used
        disp_thresh: threshold for disparity to use, values greater than this will be used
    Returns: 8 types of speed curves, M=mean, F=filter (temporal smoothing), _o=outlier pixels omitted
    TC=turning copensation, lr=left-right
        M(OF)
        M(OF)   / M(MD)
        M(OF_o)
        M(OF_o) / M(MD_o)        
        M(OF_o / MD_o)
        M(F(OF))
        M(F(OF)) / M(F(MD))
        M(F(OF) / F(MD))
        TC.M(OFlr)
        TC.M(OFlr) / M(MD)
        TC.M(OFlr / MDlr)
        TC.M(OFlr) / M(MDlr)
    '''
    hdf5_file = tables.open_file(hdf5_path, mode='r')
    h, w, _, N = hdf5_file.root.flows.shape
    print('Number of frames: ', N)
    md = np.load(npy_path)
    print('OF: ', hdf5_file.root.flows.shape)
    print('MONO: ', md.shape)
    assert md.shape[0] == N and md.shape[1] == h and md.shape[2] == w
    
    # create 3D matrix of OF, similar to Monodepth matrix 'md'
    of_mag = []
    of_hor = []
    for i in range(N):
        dx = hdf5_file.root.flows[:, :, 0, i] # horizontal OF
        dy = hdf5_file.root.flows[:, :, 1, i] # vertical OF
        of_hor.append(dx)                       # horizontal OF
        of_mag.append(np.sqrt(dx*dx + dy*dy))   # OF magnitude
    ##of = np.asarray(of_hor)
    of = np.asarray(of_mag)
    of_hor = np.asarray(of_hor)
    assert of.shape == md.shape
    hdf5_file.close()
    
    print('Applying temporal smoothing filter...')
    M_of = np.mean(of, axis=(1, 2))
    M_md = np.mean(md, axis=(1, 2))
    of_div_md = np.divide(of, md)
    if filt_mode == 'median':
        F_of = ndimage.median_filter(of, size=(filt_size, 1, 1), mode='reflect')
        F_md = ndimage.median_filter(md, size=(filt_size, 1, 1), mode='reflect')
        ##F_of_div_md = ndimage.median_filter(of_div_md, size=(filt_size, 1, 1), mode='reflect')
    elif filt_mode == 'avg':
        F_of = ndimage.filters.convolve1d(of, np.ones((filt_size,), dtype=np.float32) / float(filt_size), axis=0)
        F_md = ndimage.filters.convolve1d(md, np.ones((filt_size,), dtype=np.float32) / float(filt_size), axis=0)
        ##F_of_div_md = ndimage.filters.convolve1d(of_div_md, np.ones((filt_size,), dtype=np.float32) / float(filt_size), axis=0)
    elif filt_mode == 'gaussian':
        F_of = ndimage.filters.gaussian_filter1d(of, filt_size, axis=0)
        F_md = ndimage.filters.gaussian_filter1d(md, filt_size, axis=0)
        ##F_of_div_md = ndimage.filters.gaussian_filter1d(of_div_md, filt_size, axis=0)
    else:
        raise ValueError('Unknown temporal smoothing filter mode: %s!' % filt_mode)    
    M_F_of = np.mean(F_of, axis=(1, 2))
    
    M_of_div_md = np.mean(of_div_md, axis=(1, 2))
    ##M_F_of_div_md = np.mean(F_of_div_md, axis=(1, 2))
    M_F_of_div_F_md = np.mean(F_of / F_md, axis=(1, 2))
    M_of_div_M_md = M_of / M_md
    M_F_of_div_M_F_md = np.mean(F_of, axis=(1, 2)) / np.mean(F_md, axis=(1, 2))
    
    M_of_out = []
    M_of_div_md_out = []
    M_of_div_M_md_out = []
    TC_M_oflr = []          # turning compensation
    TC_M_oflr_M_md = []     # turning compensation
    TC_M_oflr_mdlr = []     # turning compensation
    TC_M_oflr_M_mdlr = []   # turning compensation
    for i in range(N):
        #print('Frame %d out of %d' % (i, N))
        dx = of[i, :, :]        # magnitude/horizontal flow
        disp = md[i, :, :]      # disparity
        dxo = of[i, :, :]
        #print("OF ", dx)
        #print("MD ", disp)
        #print("OF / MD ", of_div_md)
        # delete outliers (change conditions of outlier filtering)
        goodi = (disp > disp_thresh) & (dx > flow_thresh)
        if np.sum(goodi) == 0:
            dx = [0]
            disp = [1]
        else:
            dx = dx[goodi]
            disp = disp[goodi]
        
        dxo = dxo[dxo>flow_thresh]
        if np.sum(dxo)== 0:
            dxo = [0]

        # TURNING COMPENSATION
        ofhor_left = of_hor[i, :, :int(w/2)]
        ofhor_right = of_hor[i, :, int(w/2):]
        if np.mean(ofhor_left) * np.mean(ofhor_right) > 1: # left and right horizontal OF have the same sign, i.e. turning
            of_left = np.abs(of_hor[i, :, :int(w/2)])
            of_right = np.abs(of_hor[i, :, int(w/2):])
            disp_left = md[i, :, :int(w/2)]
            disp_right = md[i, :, int(w/2):]
            '''# use threshold at turning compensation too
            goodi_left = (of_left > flow_thresh) & (disp_left > disp_thresh)
            if np.sum(goodi_left) == 0:
                of_left = [0]
                disp_left = [1]
            else:
                of_left = of_left[goodi_left]
                disp_left = disp_left[goodi_left]
                
            goodi_right = (of_right > flow_thresh) & (disp_right > disp_thresh)
            if np.sum(goodi) == 0:
                of_right = [0]
                disp_right = [1]
            else:
                of_right = of_right[goodi_right]
                disp_right = disp_right[goodi_right]
            '''
            TC_M_oflr.append(abs(np.mean(of_left)-np.mean(of_right)))
            TC_M_oflr_M_md.append(abs(np.mean(of_left)-np.mean(of_right)) / np.mean(disp))
            TC_M_oflr_mdlr.append(np.abs(np.mean(np.divide(of_left, disp_left))-np.mean(np.divide(of_right, disp_right))))
            TC_M_oflr_M_mdlr.append(abs(np.mean(of_left)/np.mean(disp_left) - np.mean(of_right)/np.mean(disp_right)))
        else:
            TC_M_oflr.append(np.mean(dxo))
            TC_M_oflr_M_md.append(np.mean(dx) / np.mean(disp))
            TC_M_oflr_mdlr.append(np.mean(np.divide(dx, disp))) # same as M_of_div_md_out
            TC_M_oflr_M_mdlr.append(np.mean(dx) / np.mean(disp)) # same as M_of_div_M_md_out
        #print(i, len(goodind), np.mean(dx), np.mean(disp))
        M_of_out.append(np.mean(dxo))
        M_of_div_md_out.append(np.mean(np.divide(dx, disp)))
        M_of_div_M_md_out.append(np.mean(dx) / np.mean(disp))
    M_of_out = np.asarray(M_of_out)
    M_of_div_md_out = np.asarray(M_of_div_md_out)
    M_of_div_M_md_out = np.asarray(M_of_div_M_md_out)
    return np.vstack((M_of, M_of_div_M_md, \
        M_of_out, M_of_div_M_md_out, M_of_div_md_out, \
        M_F_of, M_F_of_div_M_F_md, M_F_of_div_F_md, 
        TC_M_oflr, TC_M_oflr_M_md, TC_M_oflr_mdlr, TC_M_oflr_M_mdlr)).T, \
        ['M(OF)', 
        'M(OF) / M(MD)', 
        'M(OF_o)', 
        'M(OF_o) / M(MD_o)',
        'M(OF_o / MD_o)', 
        'M(F(OF))', 
        'M(F(OF)) / M(F(MD))', 
        'M(F(OF) / F(MD))', 
        'TC.M(OFlr)', 'TC.M(OFlr) / M(MD)', 'TC.M(OFlr / MDlr)', 'TC.M(OFlr) / M(MDlr)']