import os
import cv2
import numpy as np
import tqdm
import pandas
import Model.Algorithms.speed.readFlowFile as readFlowFile
import math
import matplotlib.pyplot as plt
from PIL import Image
import Model.Algorithms.speed.computeColor as computeColor
import imageio
from skimage import measure
from natsort import natsorted
from scipy import ndimage

RESULTS = 'results'
OTHER_DIR = os.path.join(RESULTS, 'other')
VL_DIR = os.path.join(RESULTS, 'velocity')
NP_DIR = os.path.join(RESULTS, 'numbers')
MASK_DIR = os.path.join(RESULTS, 'mask')
DRAW_DIR = os.path.join(RESULTS, 'draw')
SUPER_PIXEL_DIR = os.path.join(RESULTS, 'super_pixel')
PLOT_SPEED_DIR = os.path.join(RESULTS, 'plot_speed')
PLOT_ERROR_DIR = os.path.join(RESULTS, 'plot_error')
PLOT_CRASH_DIR = os.path.join(RESULTS, 'plot_crash')

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351
width_to_focal[1226] = 707.0493 

fps = 10
baseline = 0.54
min_depth = 1e-3
max_depth = 80
max_velocity = 60

def getResultDirs():
    """Get dictionary with paths where the results are saved
    
    Returns:
        dictionary -- Dictionary containing paths to the results
    """
    results = {"Results": RESULTS,
    "Plot_Crash": PLOT_CRASH_DIR,
    "Plot_Error": PLOT_ERROR_DIR,
    "Plot_Speed": PLOT_SPEED_DIR,
    "Other": OTHER_DIR,
    "Numbers": NP_DIR,
    "Velocity": VL_DIR,
    "Mask": MASK_DIR,
    "Draw": DRAW_DIR, 
    "SuperPixel": SUPER_PIXEL_DIR}
    return results

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

    z = (avg_shifted_depth - avg_fst_depth) * fps * 3.6

    #v = avg_flow[:,:,0]
    #u = avg_flow[:,:,1]
    #h_angle, v_angle = calc_angle_of_view(width_to_focal[labels.shape[1]], u, v)
    h_angle, v_angle = calc_angle_of_view(width_to_focal[labels.shape[1]], avg_flow[:,:,1], avg_flow[:,:,0])

    depth = avg_fst_depth
    x, y = calc_size(h_angle, v_angle, depth)
    #x, y = (x * fps) * 3.6, (y * fps) * 3.6
    np.multiply(x, fps, out=x)
    np.multiply(x, 3.6, out=x)
    np.multiply(y, fps, out=y)
    np.multiply(y, 3.6, out=y)

    velocity[:,:,0] = y
    velocity[:,:,1] = x
    velocity[:,:,2] = z

    return velocity

def calculate_velocity_and_orientation_vectors_vectorised(of_mask, next_position, prev_position, flow, 
                                            fst_depth, snd_depth):
    """Calculate velocity based on optical flow and depth
    
    Arguments:
        of_mask {numpy array} -- optical flow mask
        next_position {numpy array} -- position after shifting by optical flow
        prev_position {numpy array} -- position before shifting by optical flow
        flow {numpy array} -- optical flow values
        fst_depth {numpy array} -- depth values on the current image
        snd_depth {numpy array} -- depth values on the shifted image
    
    Returns:
        numpy array -- velocity for each pixel
    """
    velocity = np.zeros((fst_depth.shape[0], fst_depth.shape[1], 3), dtype=np.float32)
    h,w = fst_depth.shape

    fst_mono = fst_depth
    fst_mono[of_mask] = 0

    #snd_mono_2 = np.full_like(fst_mono, 0)
    snd_mono_2 = snd_depth[prev_position[..., 0], prev_position[..., 1]]
    snd_mono_2[of_mask] = 0

    z = (snd_mono_2 - fst_mono) * fps * 3.6

    
    v = flow[:,:,0]
    u = flow[:,:,1]
    h_angle, v_angle = calc_angle_of_view(width_to_focal[fst_depth.shape[1]], u, v)
    depth = fst_depth
    x, y = calc_size(h_angle, v_angle, depth)
    x, y = (x * fps) * 3.6, (y * fps) * 3.6

    velocity[:,:,0] = y
    velocity[:,:,1] = x
    velocity[:,:,2] = z
    return velocity


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
    #depth = np.multiply(width_to_focal[width], np.divide(baseline, np.multiply(width, disparity)))
    np.multiply(width_to_focal[width], np.divide(baseline, np.multiply(width, disparity)), out=disparity)
    
    return disparity

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
    return fns

def vector_distance(x,y,z):
    """Calculate the euclidean distance of the coordinates 
    
    Arguments:
        x {float} -- x coordinate
        y {float} -- y coordinate
        z {float} -- z coordinate
    
    Returns:
        float -- the euclidean distance
    """
    x2 = np.power(x, 2, dtype=np.float32)
    y2 = np.power(y, 2, dtype=np.float32)
    z2 = np.power(z, 2, dtype=np.float32)
    #added = x2 + y2 + z2
    return np.sqrt(x2 + y2 + z2)

def reduce_sort(vector, low=0.1,high=0.9, skip=None):
    """Drop values from the given vector
    
    Arguments:
        vector {numpy array} -- the vector to drop values from
    
    Keyword Arguments:
        low {float} -- after sorting drop this percent from the bottom (default: {0.1})
        high {float} -- after sorting drop this percent from the top (default: {0.9})
        skip {float} -- skip values with this value (default: {None})
    
    Returns:
        numpy array -- the reduced vector
    """
    v_unique = np.sort(vector.flatten())
    if skip is not None:
        v_unique = v_unique[v_unique != skip]

    mask_vector = np.logical_and(vector >= v_unique[int(len(v_unique) * low)],
        vector <= v_unique[int(len(v_unique) * high) - 1])
    return mask_vector


def vector_speedOF_Simple(vectors, high=1, low=0, optimal_of=True):
    """Estimate speed from the x,y vectors
    
    Arguments:
        vectors {numpy array} -- vector containing x, y and z
    
    Keyword Arguments:
        low {float} -- after sorting drop this percent from the bottom
        high {float} -- after sorting drop this percent from the top 
    
    Returns:
        (float, numpy array) -- Tuple: speed estimation, mask used for the estimation
    """
    x = vectors[:,:,0]
    y = vectors[:,:,1]
    mask_uni = reduce_sort(y,low=low,high=high)
    print(low, high)
    if optimal_of:
        only_good_of = x != 0
        #mask_uni = np.logical_and(mask_uni, only_good_of)
        np.logical_and(mask_uni, only_good_of, out=mask_uni)


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
        
        return TC, mask_uni

    speed = np.average([np.mean(x_abs), np.mean(y_abs)])

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

def average(data, labels, index, masks):
    """Return data with values replaced by the mean value of the superpixel values
    
    Arguments:
        data {numpy array} -- matrix (usually from an image)
        labels {numpy array} -- superpixel labels
    
    Returns:
        numpy array -- same shape as data with values replaced by the mean value of the superpixel values
    """
    avg = np.zeros_like(data, dtype=np.int16)
    regions = ndimage.mean(data, labels=labels, index=index)
    for i, reg in enumerate(regions):
        avg[masks[i]] = reg
    return avg

def error_comparison_Speed_Vecors(speed_est, speed_gt, csv=None, visualize=True):
    """Visualize error between estimated speed and the ground truth speed
    
    Args:
        speed_est (float): [The estimated speed]
        speed_gt (float): [Ground truth speed]
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
        
        frame_mask = mask[i] # 375 x 1242
        
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
        cv2.putText(z, 'Frame ' + str(i), (50, 450), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, 'Speed estimation', (50, 480), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, 'GT:   ' + str(round(speed_gt[i], 3)), (50, 510), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, ' simple: ' + str(np.round(speed_simple[i], 3)), (50, 570), font, fSize, (255, 255, 255), fThick)
        cv2.putText(z, '    err: ' + str(round(speed_gt[i] - speed_simple[i], 3)), (220, 570), font, fSize, (255, 255, 255), fThick)
        
        vid_out.writeFrame(cv2.cvtColor(z, cv2.COLOR_BGR2RGB))
        

