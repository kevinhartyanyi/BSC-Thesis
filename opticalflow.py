import cv2
import sys
import os
import numpy as np
import tables         ## !!!On Windows: You need to build zlib from source with CMake and Visual Studio (https://www.zlib.net/)
import skvideo.io




def create_flow_box_visualization(flowbox, brightness_mul=10.):
    """
    Creates a flowbox visualization, where for each pixel, brightness represents the flow vector
        distance and the color its angle (projected to the hue in the HSV coding).
    Parameters:
        flowbox: ndarray(size_y, size_x, 2:[y,x]) of float32; output of optical flow algorithms
        brightness_mul: float;
    Returns:
        flow3box: ndarray(size_y, size_x, 3) of uint8
    """
    assert flowbox.ndim == 3
    assert flowbox.shape[2] == 2
    MAX_HUE = 179.
    flow3box = np.empty((flowbox.shape[0], flowbox.shape[1], 3), dtype=np.uint32)
    flow3box.fill(255)
    r, phi = cart2polar(flowbox[:, :, 1], flowbox[:, :, 0])
    flow3box[:, :, 0] = ((phi + np.pi) / (2. * np.pi) * MAX_HUE).astype(np.uint32)
    flow3box[:, :, 2] = (r * brightness_mul).astype(np.uint32)
    flow3box[:, :, 1:] = np.clip(flow3box[:, :, 1:], 0, 255)
    flow3box[:, :, 0] = np.clip(flow3box[:, :, 0], 0, int(MAX_HUE))
    flow3box = flow3box.astype(np.uint8)
    flow3box = cv2.cvtColor(flow3box, cv2.COLOR_HSV2BGR)
    return flow3box


def cart2polar(x, y):
    """
    Elementwise Cartesian2Polar for arrays. x and y should be of the same size.
    Parameters:
        x, y: ndarray(...); cartesian coordinate arrays
    Returns:
        r: ndarray(...); radius
        phi: ndarray(...); angle in radians, -pi..pi
    """
    r = np.sqrt(np.square(x) + np.square(y))
    phi = np.arctan2(y, x)
    return r, phi

def run_optflow(vid_path, out_path, of_type = 'dense'):
    '''
    Runs OF on video: output is saved in .HDF5 file and is also visualized.
    Input parameters:
        vid_path = full path of input video
        out_path = full path of output visualized video
        of_type = 'dense' or 'flownet2'
    '''
    if not os.path.exists(vid_path):
        raise ValueError('Input video file does not exist.')
    
    if of_type == 'dense':
        of = None
    else:
        metadata = skvideo.io.ffprobe(vid_path)
        w = int(metadata['video']['@width'])
        h = int(metadata['video']['@height'])
        # size of FlowNet2 input must be divisible by 64
        wFlow = w + (64 - (w % 64)) % 64     # increase it to be divisible by 64
        hFlow = h + (64 - (h % 64)) % 64
        of = FlowNet2(boxsize = (hFlow, wFlow), model_path = CAFFE_MODEL, prototxt_path = CAFFE_DEPLOY_PROTOTXT)
    
    optical_flow(vid_path, out_path, of = of)

def optical_flow(vid_path, out_path, of = None):
    """ 
    Runs optical flow on video, saves result in .hdf5 files and visualizes it on video.
    Input parameters:
        vid_path = full path of input video
        out_path = full path of output visualized video
        of == FlowNet2 class from nipgutils.optflow_interface - Optical Flow with Deep Networks
        of == None - OpenCV's Dense Optical Flow
    """
    vid = cv2.VideoCapture(vid_path)
    if not vid.isOpened():
        error('Could not open video!')
    frameNr = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # get HSV representation of OF vector
    n = 50 # change to smaller value if video resolution small
    of_hsv = of_vector_hsv(n)
    vid_out = skvideo.io.FFmpegWriter(out_path)
    
    hdf5_file = tables.open_file(out_path[:-4] + '.hdf5', mode='w')
    flows = hdf5_file.create_earray(hdf5_file.root, 'flows', 
                                    atom = tables.Atom.from_dtype(np.dtype('Float32')), 
                                    shape = (h, w, 2, 0), 
                                    filters = tables.Filters(complevel=5, complib='blosc'), 
                                    expectedrows = frameNr-1)
    _, frame = vid.read()
    f1 = frame
    for i in range(1, frameNr):
        if i % 100 == 0:
            print('Frame %d out of %d' % (i, frameNr))
        _, frame = vid.read()
        f2 = frame
        
        if of is None:
            flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY), 
                    flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
            flow = np.flip(flow, axis = 2)
        else:
            flow = run_flownet2(f1, f2, of, w, h)
        
        flows.append(np.expand_dims(flow, axis=3))
        v = create_flow_box_visualization(flow)
        #v = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
        v[5:5+n, 5:5+n] = of_hsv
        if i == 1: # repeat first OF for first frame
            flows.append(np.expand_dims(flow, axis=3))
            conc = np.concatenate((cv2.cvtColor(f1, cv2.COLOR_BGR2RGB), v))
            vid_out.writeFrame(conc)
            
        conc = np.concatenate((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), v))
        vid_out.writeFrame(conc)
        f1 = f2
    vid_out.close()
    hdf5_file.close()


def of_vector_hsv(n = 50):
    '''
    Generates HSV representation of OF vector movements.
    Input parameters:
        n = width and height of output square matrix
    '''
    dx = np.zeros((n, n))
    dy = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            dx[i, j] = j
            dy[i, j] = i
    dx -= n/2
    dy -= n/2
    '''
    # uncomment for circle (instead of square
    for i in range(n):
        for j in range(n):
            if (i - n/2)*(i - n/2) + (j - n/2)*(j - n/2) > n*n/4.0:
                dx[i, j] = 0
                dy[i, j] = 0
    '''
    flow = np.dstack((dy, dx))
    
    v = create_flow_box_visualization(flow)
    #v = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
    #imageio.imwrite('of.png', v)
    return v

def of_visualize(hdf5_path, out_path):
    '''
    Visualizes previously saved OF results.
    Input parameters:
        hdf5_path = .hdf5 file containing OF results (is output of run_optflow())
        out_path = full path of output visualized video
    '''
    hdf5_file = tables.open_file(hdf5_path, mode='r')
    flows = hdf5_file.root.flows[:]
    d = flows.shape
    h, w, N = d[0], d[1], d[3]
    vid_out = skvideo.io.FFmpegWriter(out_path, outputdict={
        '-vcodec': 'libx264',
        '-pix_fmt': 'yuv420p',
        '-s': (str(w)+'x'+str(h))
    })
    # get HSV representation of OF vector
    n = 50
    of_hsv = of_vector_hsv(n)

    for i in range(N):
        if i % 100 == 0:
            print('Frame %d out of %d' % (i, N))
        flow = flows[:, :, :, i]
        v = create_flow_box_visualization(flow)
        #v = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
        v[5:5+n, 5:5+n] = of_hsv
        vid_out.writeFrame(v)
    hdf5_file.close()
    vid_out.close()


run_optflow("b.mp4", "/home/kevin/t.mp4")