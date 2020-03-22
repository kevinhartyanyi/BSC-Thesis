import argparse
import itertools
import multiprocessing
import os
import glob
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.color import rgb2gray
from skimage.filters import sobel
import cv2
import numpy as np
import tqdm

from natsort import natsorted # Run with python3

import speed.readFlowFile as readFlowFile
import speed.computeColor as computeColor

import speed.utils as utils

import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt

KITTI_PATH = '/home/kevin/Programming/School/semester_6/BSC-Thesis/src/main/resources/base/data'

class Video():
    def __init__(self, video_number = '0001', crop = False):
        self.video_number = video_number
        self.images = None
        self.pwc = None
        self.back_pwc = None
        self.mono = None
        self.video = None
        self.gt = None
        self.found = False
        self.pwc_video = False
        self.back_pwc_video = False
        self.mono_video = False
        self.hd3_video = False
        self.hd3 = False
        self.back_hd3 = False
        self.back_hd3_video = False
        folder = self.__load()

        
    def __load(self):
        kitti_dirs = os.listdir(KITTI_PATH)
    
        for i, video in enumerate(kitti_dirs):
            base, number = video.rsplit('_', 1)
            print(video, self.video_number)
            if number != self.video_number:
                continue

            
            self.video = KITTI_PATH + '/' + video       + '/' + video + '.mp4'       #+ '/test'
            self.images = KITTI_PATH + '/' + video      + '/Pictures/'      #+ '/test'
            self.pwc = KITTI_PATH + '/' + video         + '/Pwc/'      #+ '/test'
            self.back_pwc = KITTI_PATH + '/' + video    + '/Back_Pwc/'      #+ '/test'
            self.mono = KITTI_PATH + '/' + video        + '/Mono/'      #+ '/test'
            self.gt = np.load(KITTI_PATH + '/' + video  + '/speed_gt.npy')      #+ '/test'            
            
            pwc_vid = KITTI_PATH + '/' + video        + '/pwc' + '.mp4'
            back_pwc_vid = KITTI_PATH + '/' + video   + '/back_pwc' + '.mp4'
            mono_vid =  KITTI_PATH + '/' + video      + '/mono' + '.mp4'
            
            if os.path.exists(pwc_vid) is False:
                utils.create_video_of_or_mono(pwc_vid, self.pwc)
            if os.path.exists(back_pwc_vid) is False:
                utils.create_video_of_or_mono(back_pwc_vid, self.back_pwc)
            if os.path.exists(mono_vid) is False:
                utils.create_video_of_or_mono(mono_vid, self.mono, mono=True)

            self.pwc_video = pwc_vid
            self.back_pwc_video = back_pwc_vid
            self.mono_video = mono_vid

            self.hd3_video = KITTI_PATH + '/' + video        + '/HD3/' + 'HD3.avi'
            self.hd3 = KITTI_PATH + '/' + video              + '/HD3/' 
            self.back_hd3 = KITTI_PATH + '/' + video         + '/Back_HD3/'
            self.back_hd3_video = KITTI_PATH + '/' + video   + '/Back_HD3/' + 'HD3.avi'

            return video

        if self.pwc is None:
            raise ValueError('Wrong video number or missing pwc')
        if self.back_pwc is None:
            raise ValueError('Wrong video number or missing back_pwc')

OUT_PATH = '/home/kevin/Programming/School/semester_6/BSC-Thesis/'

RESULTS = 'results'
OTHER_DIR = os.path.join(RESULTS, 'other')
VL_DIR = os.path.join(RESULTS, 'velocity')
NP_DIR = os.path.join(RESULTS, 'numbers')
MASK_DIR = os.path.join(RESULTS, 'mask')
DRAW_DIR = os.path.join(RESULTS, 'draw')
SUPER_PIXEL_DIR = os.path.join(RESULTS, 'super_pixel')


DATA = '/home/kevin/workspace/pipeline_zero/test2/'

DATA_TRUE = '/home/kevin/workspace/bosch_velocity/data/KITTI/data_scene_flow'
DISP1 = DATA_TRUE + '/training/disp_occ_0'
DISP2 = DATA_TRUE + '/training/disp_occ_1'
FLOW_DIR = DATA_TRUE + '/training/flow_occ'
IMAGES = DATA_TRUE + '/training/image_2'

DATA_TEST = '/home/kevin/workspace/bosch_velocity'
TEST = 'training_2048'

#BORUVKA_OUT_TRAIN = DATA_TEST + '/data/boruvka_out/' + TEST

BORUVKA_OUT_TRAIN = '/home/kevin/Programming/School/semester_6/BSC-Thesis/src/main/resources/base/results'

SLIC_OUT_TRAIN = DATA_TEST + '/data/slic_out/' + TEST
PWC_OUT_TRAIN = DATA_TEST + '/data/pwc_out/training'
MD_OUT_TRAIN = DATA_TEST + '/data/md_out/training'
BACKOF = DATA_TEST + '/data/pwc_out/backOF'

SINGLE_VIDEO =   KITTI_PATH
IMAGES_SINGLE =  KITTI_PATH + '/2011_09_26_drive_0001/test/Pictures/'
MONO_SINGLE =    KITTI_PATH + '/2011_09_26_drive_0001/test/Mono/'
PWC_SINGLE =     KITTI_PATH + '/2011_09_26_drive_0001/test/Pwc/'
BACKPWC_SINGLE = KITTI_PATH + '/2011_09_26_drive_0001/test/Back_Pwc/'

VIDEOS = []
TRAIN_VIDEOS = ['0001', '0002', '0005', '0009', '0014', '0019', '0027']


def run(img_dir, depth_dir, of_dir, of_back_dir, save_dir, speed_gt=None, high=1, low=0.309, super_pixel_method=""):

    
    main(super_pixel_method, img_dir=img_dir, disp_dir=depth_dir, disp2_dir=None, test_number = 6, back_flow_dir=of_back_dir, flow_dir=of_dir, out_dir=save_dir, label_dir=img_dir,
    high=high, low=low, visualize = True)
    
    
    #speeds_dir = utils.list_directory(OUT_PATH, extension='speed.npy')
    #speeds_dir = natsorted(speeds_dir)
    #speeds = []
    #for s in speeds_dir:
    #    speeds.append(np.load(s))
    #
    #speeds_mask = utils.list_directory(OUT_PATH, extension='mask.npy')
    #speeds_mask = natsorted(speeds_mask)
    #masks = []
#
    #for s in speeds_mask:
    #    masks.append(np.load(s))
#
    #print(len(speeds))
    #print(speed_gt.shape)
    #_ = utils.error_comparison_Speed_Vecors(speeds,speed_gt[1:],csv=OUT_PATH+str(vid_id)+'_error_Simple_OF.csv')
    #if flow == "pwc":
    #    utils.create_speed_video_Speed_Vectors(vid.video, str(vid_id)+'_video.mp4', speeds, speed_gt, masks,
    #            vid.pwc_video, vid.back_pwc_video, vid.mono_video)
    #else:
    #    utils.create_speed_video_Speed_Vectors(vid.video, str(vid_id)+'_video.mp4', speeds, speed_gt, masks,
    #            vid.hd3_video, vid.back_hd3_video, vid.mono_video, hd3=True)
    #
    ##video = Video("0001")
    ##main(img_dir= DATA + 'origin', disp_dir=DATA + 'monodepth', flow_dir=DATA + 'pwc', label_dir=DATA + 'boruvka', visualize = True)
    
    #main(img_dir= IMAGES, disp_dir=DISP1, disp2_dir=DISP2, flow_dir=FLOW_DIR, label_dir=SLIC_OUT_TRAIN, visualize = True, back_flow_dir=None)
    #main(img_dir= IMAGES, disp_dir=MD_OUT_TRAIN, disp2_dir=None, test_number = 2, back_flow_dir=BACKOF, flow_dir=PWC_OUT_TRAIN, label_dir=BORUVKA_OUT_TRAIN, visualize = True)

    #vid = Video(VIDEOS[0])

    
    #main(img_dir= IMAGES_SINGLE, disp_dir=MONO_SINGLE, disp2_dir=None, test_number = 6, back_flow_dir=BACKPWC_SINGLE, flow_dir=PWC_SINGLE, label_dir=BORUVKA_OUT_TRAIN, visualize = True)
    """
    check = []

    best_avg = 100
    best_low = 0.5
    step_size = 0.5

    check_high = []
    best_high = 0.5
    step_size_high = 0.5

    
    f = open("optimise_results_hd3_v2.txt", "w")
    stopcount = 3
    count = 0
    while(True):
        check, step_size = utils.gen_numbers(best_low, step_size)
        check_high, step_size_high = utils.gen_numbers(best_high, step_size_high)
        print("Iteration: ", count)
        print(check)
        print(check_high)
        for c_h in check_high:
            if c_h <= 0.2:
                continue
            for c in check:
                if c >= 0.8 or c >= c_h:
                    continue
                errors = []
                for vid_id in TRAIN_VIDEOS:
                    vid = Video(vid_id)

                    speeds_dir = utils.list_directory(OUT_PATH, extension='speed.npy')
                    for s in speeds_dir:
                        os.remove(s)

                    main(img_dir= vid.images, disp_dir=vid.mono, disp2_dir=None, test_number = 6, back_flow_dir=vid.back_hd3, flow_dir=vid.hd3, label_dir=BORUVKA_OUT_TRAIN,
                        high=c_h,low=c)

                    #main(img_dir= vid.images, disp_dir=vid.mono, disp2_dir=None, test_number = 6, back_flow_dir=vid.back_pwc, flow_dir=vid.pwc, label_dir=BORUVKA_OUT_TRAIN,
                    #    high=1,low=c)
                    
                    speed_gt = vid.gt
                    
                    speeds_dir = utils.list_directory(OUT_PATH, extension='speed.npy')
                    speeds_dir = natsorted(speeds_dir)
                    speeds = []
                    for s in speeds_dir:
                        speeds.append(np.load(s))
                    rmse = utils.error_comparison_Speed_Vecors(speeds,speed_gt[1:],csv=OUT_PATH+str(vid_id)+'_error_Simple_OF.csv', visualize=False)
                    errors.append(rmse)

                    for s in speeds_dir:
                        os.remove(s)
                avg = sum(errors)/len(errors)
                f.write("," + str(avg) + "," + str(c) + "," + str(c_h) + ",\n")
                print("Avg: " + str(avg) + " Low: " + str(c) + " High: " + str(c_h))
                if avg < best_avg:
                    best_avg = avg
                    best_low = c
                    best_high = c_h
                    f.write("Best," + str(best_avg) + "," + str(best_low) + "," + str(best_high) + ",\n")
        count += 1
        if count >= stopcount:
            break          
    f.close()
    assert 1 == 2
    """
    
    """#for vid_id in TRAIN_VIDEOS:
    #    vid = Video(vid_id)
    vid_id = '0001'
    vid_name = '/' + vid_id
    vid = Video(vid_id)
    #utils.concat_video3(vid.video, vid.pwc_video, vid.hd3_video, "pwc_hd3.mp4")
    flow = "pwc"

    speeds_dir = utils.list_directory(OUT_PATH, extension='speed.npy')
    mask_dir = utils.list_directory(OUT_PATH, extension='mask.npy')
    for s in speeds_dir:
        os.remove(s)
    for s in mask_dir:
        os.remove(s)

    if flow == "pwc":
        main(vid_name, img_dir= vid.images, disp_dir=vid.mono, disp2_dir=None, test_number = 6, back_flow_dir=vid.back_pwc, flow_dir=vid.pwc, label_dir=BORUVKA_OUT_TRAIN,
        high=1, low=0.309, visualize = True)
    else:
        main(vid_name, img_dir= vid.images, disp_dir=vid.mono, disp2_dir=None, test_number = 6, back_flow_dir=vid.back_hd3, flow_dir=vid.hd3, label_dir=BORUVKA_OUT_TRAIN,
        high=1, low=0.36, visualize = True)
    print(vid.pwc_video)
    speed_gt = vid.gt
    print("GT: " + str(speed_gt[0]))
    
    speeds_dir = utils.list_directory(OUT_PATH, extension='speed.npy')
    speeds_dir = natsorted(speeds_dir)
    speeds = []
    for s in speeds_dir:
        speeds.append(np.load(s))
    
    speeds_mask = utils.list_directory(OUT_PATH, extension='mask.npy')
    speeds_mask = natsorted(speeds_mask)
    masks = []

    for s in speeds_mask:
        masks.append(np.load(s))

    print(len(speeds))
    print(speed_gt.shape)
    _ = utils.error_comparison_Speed_Vecors(speeds,speed_gt[1:],csv=OUT_PATH+str(vid_id)+'_error_Simple_OF.csv')
    if flow == "pwc":
        utils.create_speed_video_Speed_Vectors(vid.video, str(vid_id)+'_video.mp4', speeds, speed_gt, masks,
                vid.pwc_video, vid.back_pwc_video, vid.mono_video)
    else:
        utils.create_speed_video_Speed_Vectors(vid.video, str(vid_id)+'_video.mp4', speeds, speed_gt, masks,
                vid.hd3_video, vid.back_hd3_video, vid.mono_video, hd3=True)
    
    for s in speeds_dir:
        os.remove(s)
    for s in speeds_mask:
        os.remove(s)"""
    
    #speed = np.load(OUT_PATH + '37' + '_speed.npy')
    #print("GT: " + str(video.gt[36]))
    #print("Our:" + str(np.mean(speed)))

class VelocityCalculator(object):
    def __init__(self,fst_img_fn, snd_img_fn, fst_depth_fn, snd_depth_fn, 
                label_fn, flow_fn, back_flow, out_dir, use_slic, n_sps, visualize_results=True, high=1, low=0, super_pixel_method=""):
        self.read_depth = utils.read_depth
        self.read_flow = readFlowFile.read
        self.average = utils.average
        self.calculate_shifted_labels = utils.calculate_shifted_labels
        
        self.fst_img_fn = fst_img_fn
        self.snd_img_fn = snd_img_fn
        self.fst_depth_fn = fst_depth_fn
        self.snd_depth_fn = snd_depth_fn
        self.label_fn = label_fn
        self.flow_fn = flow_fn
        self.out_dir = out_dir
        self.use_slic = use_slic
        self.n_sps = n_sps
        self.visualize_results = visualize_results
        self.back_flow = back_flow
        self.high = high
        self.low = low
        #self.vid_name = vid_name
        self.super_pixel_method = super_pixel_method


    def calculate_velocity_and_orientation(self):
        if self.back_flow is "NA":
            # Read left and right images
            fst_img = cv2.imread(self.fst_img_fn, cv2.IMREAD_COLOR)  # height x width x channels
            snd_img = cv2.imread(self.snd_img_fn, cv2.IMREAD_COLOR)  # height x width x channels

            assert fst_img.shape == snd_img.shape
            height, width, _ = fst_img.shape
            
            # Read disparity maps
            fst_depth = self.read_depth(self.fst_depth_fn, width, height)
            snd_depth = self.read_depth(self.snd_depth_fn, width, height)

            flow = self.read_flow(self.flow_fn)

            base_fn = os.path.join(self.out_dir, 
                                os.path.splitext(os.path.basename(self.flow_fn))[0])

            # Calculate the velocity and the orientation
            velocity = \
                utils.calculate_velocity_and_orientation_vectors_vectorised_gt(
                                                                flow, 
                                                                fst_depth, 
                                                                snd_depth)
            

            # Save the results
            base_fn = os.path.join(self.out_dir, 
                                os.path.splitext(os.path.basename(self.flow_fn))[0])
            np.save(base_fn + '_vel.npy', velocity)
            #np.save(base_fn + '_ori.npy', orientation)
            speed, speed_mask = utils.vector_speed(velocity, 0) # Speed calculation
            np.save(base_fn + '_speed.npy', speed)

            # Visualize the results if needed
            if self.visualize_results:
                #velocity[velocity > utils.max_velocity] = utils.max_velocity
                #velocity[velocity < -utils.max_velocity] = -utils.max_velocity
                
                # Save speed vectors as image
                masked_speed = speed
                masked_speed[~speed_mask] = 0
                utils.save_as_image(base_fn + '_speed.png', speed, min_val=0, max_val=utils.max_depth) 
                utils.save_as_image(base_fn + '_speed_masked.png', masked_speed, min_val=0, max_val=utils.max_depth) 
                # Save velocity vectors as image
                for idx, file_id in enumerate(['x', 'y', 'z']):
                    utils.save_as_image('{}_{}.png'.format(base_fn, file_id), 
                                        velocity[:, :, idx], max_val=utils.max_velocity)
                # Save depth vectors as image
                for file_id, data in zip(['d1', 'd2'],
                                        [fst_depth, snd_depth]):
                    utils.save_as_image('{}_{}.png'.format(base_fn, file_id), 
                                        data, min_val=0, max_val=utils.max_depth)
                # Save optical flow as image
                cv2.imwrite(base_fn + '_flow.png', computeColor.computeImg(flow))
                # Save the final image
                visualize(base_fn + '_viz.png', speed.astype('uint8'), velocity, base_fn + '_flow.png', 
                        fst_img, snd_img, base_fn + '_d1.png', base_fn + '_d2.png', 
                        base_fn + '_speed.png')
        
        if False and self.super_pixel_method != "":
            # Read superpixel labels
            #if self.use_slic:
            #    labels = utils.read_gslic_labels(self.label_fn, n_sps=self.n_sps)
            #else:
            #    labels = utils.read_boruvka_labels(self.label_fn, n_sps=self.n_sps)
            

            # Read left and right images
            fst_img = cv2.imread(self.fst_img_fn, cv2.IMREAD_COLOR)  # height x width x channels
            snd_img = cv2.imread(self.snd_img_fn, cv2.IMREAD_COLOR)  # height x width x channels

            if self.super_pixel_method == "Felzenszwalb":
                labels = felzenszwalb(fst_img, scale=100, sigma=0.5, min_size=50)                
            elif self.super_pixel_method == "Quickshift":
                labels = quickshift(fst_img, kernel_size=3, max_dist=6, ratio=0.5)
            elif self.super_pixel_method == "Slic":
                labels = slic(fst_img, n_segments=250, compactness=10, sigma=1)
            elif self.super_pixel_method == "Watershed":
                gradient = sobel(rgb2gray(fst_img))
                labels = watershed(gradient, markers=250, compactness=0.001)

            assert fst_img.shape == snd_img.shape
            height, width, _ = fst_img.shape
            
            # Read disparity maps
            fst_depth = self.read_depth(self.fst_depth_fn, width, height)
            avg_fst_depth = self.average(fst_depth, labels)
            snd_depth = self.read_depth(self.snd_depth_fn, width, height)
            
            # Read optical flow
            flow = self.read_flow(self.flow_fn)
            avg_flow = np.zeros_like(flow) 
            avg_flow[:, :, 0] = self.average(flow[:, :, 0], labels)
            avg_flow[:, :, 1] = self.average(flow[:, :, 1], labels)

            # Shift labels and depth values respect to the average optical flow
            shifted_labels = self.calculate_shifted_labels(labels, avg_flow)
            avg_shifted_depth = self.average(snd_depth, shifted_labels)

            # Calculate the velocity and the orientation
            velocity, orientation = \
                utils.calculate_velocity_and_orientation_vectors(labels, shifted_labels, 
                                                                avg_flow, 
                                                                avg_fst_depth, 
                                                                avg_shifted_depth)

            speed, speed_mask = utils.vector_speed(velocity, 0) # Speed calculation

            # Save the results
            base_fn = os.path.join(self.out_dir, 
                                os.path.splitext(os.path.basename(self.flow_fn))[0])
            np.save(base_fn + '_vel.npy', velocity)
            np.save(base_fn + '_vel.npy', velocity)

            x = velocity[:,:,0]
            y = velocity[:,:,1]
            z = velocity[:,:,2]
            speed_superpixel = utils.vector_distance(x,y,z)
            np.save(base_fn + '_superpixel.npy', speed_superpixel)
            

            # Visualize the results if needed
            #if self.visualize_results:
            #    velocity[velocity > utils.max_velocity] = utils.max_velocity
            #    velocity[velocity < -utils.max_velocity] = -utils.max_velocity
            #    # Save velocity vectors as image
            #    for idx, file_id in enumerate(['x', 'y', 'z']):
            #        utils.save_as_image('{}_{}.png'.format(base_fn, file_id), 
            #                            velocity[:, :, idx], max_val=utils.max_velocity)
            #    # Save depth vectors as image
            #    for file_id, data in zip(['d1', 'd2', 'd1_avg', 'd2_avg'],
            #                            [fst_depth, snd_depth, 
            #                            avg_fst_depth, avg_fst_depth]):
            #        utils.save_as_image('{}_{}.png'.format(base_fn, file_id), 
            #                            data, min_val=utils.min_depth, max_val=utils.max_depth)
            #    # Speed
            #    utils.save_as_image(base_fn + '_speed.png', speed, min_val=0, max_val=utils.max_depth)  
            #    # Save optical flow as image
            #    cv2.imwrite(base_fn + '_flow.png', computeColor.computeImg(flow))
            #    # Save the final image
            #    visualize(base_fn + '_viz.png', labels, velocity, base_fn + '_z.png', 
            #            fst_img, snd_img, base_fn + '_d1.png', base_fn + '_d2.png', 
            #            base_fn + '_flow.png')
        else:
            # Read left and right images
            fst_img = cv2.imread(self.fst_img_fn, cv2.IMREAD_COLOR)  # height x width x channels
            snd_img = cv2.imread(self.snd_img_fn, cv2.IMREAD_COLOR)  # height x width x channels

            assert fst_img.shape == snd_img.shape
            height, width, _ = fst_img.shape
            
            # Read disparity maps
            fst_depth = self.read_depth(self.fst_depth_fn, width, height)
            #avg_fst_depth = self.average(fst_depth, labels)
            snd_depth = self.read_depth(self.snd_depth_fn, width, height)
            # Read optical flow
            flow = self.read_flow(self.flow_fn)
            back_flow = self.read_flow(self.back_flow)            

            of_mask, next_position, prev_position = utils.calc_bidi_errormap(flow, back_flow, tau=0.8)
            good_flow = flow.copy()
            good_flow_snd = flow.copy()
            good_flow[of_mask] = 0
            good_flow_snd[next_position[..., 0], next_position[..., 1]] = back_flow

            fst_mono = fst_depth
            fst_mono[of_mask] = 0

            snd_depth_for_mono = snd_depth.copy()
            snd_depth_for_mono[of_mask] = 0
            snd_mono = np.full_like(fst_mono, 0)
            snd_mono[next_position[..., 0], next_position[..., 1]] = snd_depth_for_mono

            snd_mono_2 = np.full_like(fst_mono, 0)
            snd_mono_2 = snd_depth[prev_position[..., 0], prev_position[..., 1]]
            snd_mono_2[of_mask] = 0

            back = np.full_like(fst_mono, 0)
            back[prev_position[..., 0], prev_position[..., 1]] = snd_mono

            incons_img = np.tile(255*of_mask[..., None], (1, 1, 3)).astype(np.uint8)
            incons_img = utils.draw_velocity_vectors(incons_img, next_position, relative_disp=False, color=(0, 0, 255))
            
            tmp = snd_mono - fst_mono

            fst_zeros = fst_mono.copy()
            fst_zeros[snd_mono == 0] = 0
            

            # Save the results
            base_fn = self.out_dir# + self.vid_name
            img_num = os.path.splitext(os.path.basename(self.flow_fn))[0]

            # Calculate the velocity and the orientation
            velocity = \
                utils.calculate_velocity_and_orientation_vectors_vectorised(of_mask,next_position, prev_position, 
                                                                flow, 
                                                                fst_depth, 
                                                                snd_depth)
            image = velocity.copy()
            plt.imsave(os.path.join(base_fn, VL_DIR, img_num + '_velocity.png'), image.astype('uint8'))
            #print("")
            #print(np.sort(np.unique(fst_depth)))
            #print("")

            #inaccurate_mono = fst_depth > 115
            inaccurate_mono = fst_depth > 30
            fst_depth[inaccurate_mono] = 0
            velocity[inaccurate_mono] = 0

            
            #np.save(base_fn + '_vel.npy', velocity)
            #np.save(base_fn + '_ori.npy', orientation)
            plt.imsave(os.path.join(base_fn, DRAW_DIR, img_num + '_draw.png'), incons_img.astype('uint8'))

            if self.super_pixel_method != "":
                if self.super_pixel_method == "Felzenszwalb":
                    labels = felzenszwalb(fst_img, scale=100, sigma=0.5, min_size=50)                
                elif self.super_pixel_method == "Quickshift":
                    labels = quickshift(fst_img, kernel_size=3, max_dist=6, ratio=0.5)
                elif self.super_pixel_method == "Slic":
                    labels = slic(fst_img, n_segments=250, compactness=10, sigma=1)
                elif self.super_pixel_method == "Watershed":
                    gradient = sobel(rgb2gray(fst_img))
                    labels = watershed(gradient, markers=250, compactness=0.001)
                # Read disparity maps
                avg_fst_depth = self.average(fst_depth, labels)
                
                # Read optical flow
                avg_flow = np.zeros_like(flow) 
                avg_flow[:, :, 0] = self.average(flow[:, :, 0], labels)
                avg_flow[:, :, 1] = self.average(flow[:, :, 1], labels)

                # Shift labels and depth values respect to the average optical flow
                shifted_labels = self.calculate_shifted_labels(labels, avg_flow)
                avg_shifted_depth = self.average(snd_depth, shifted_labels)

                # Calculate the velocity and the orientation
                velocity_2, _ = \
                    utils.calculate_velocity_and_orientation_vectors(labels, shifted_labels, 
                                                                    avg_flow, 
                                                                    avg_fst_depth, 
                                                                    avg_shifted_depth)

                x = velocity_2[:,:,0]
                y = velocity_2[:,:,1]
                z = velocity_2[:,:,2]
                speed_superpixel = utils.vector_distance(x,y,z)
                np.save(os.path.join(base_fn, NP_DIR, img_num + '_superpixel.npy'), speed_superpixel)
                plt.matshow(speed_superpixel)
                plt.colorbar()
                plt.savefig(os.path.join(base_fn, SUPER_PIXEL_DIR, "{0}_superpixel.png".format(img_num)), bbox_inches='tight', dpi=150)

            #speed, speed_mask = utils.vector_speed(velocity, 0) # Speed calculation
            
            #speed, speed_mask = utils.vector_speedOF(velocity, 0) # Speed calculation
            #speed, speed_mask = utils.vector_speedOF4Side(velocity, 0) # Speed calculation
            #print("Low {0}, High {1}".format(self.low, self.high))
            speed, speed_mask = utils.vector_speedOF_Simple(velocity,low=self.low,high=self.high) # Speed calculation
            np.save(os.path.join(base_fn, NP_DIR, img_num + '_speed.npy'), speed)
            np.save(os.path.join(base_fn, NP_DIR, img_num + '_mask.npy'), speed_mask)

            #image = velocity.copy()
            ##image = image + abs(image.min())
            ##image *= 255.0/image.max()
            #plt.imsave(os.path.join(base_fn, VL_DIR, img_num + '_velocity.png'), image.astype('uint8'))
            #utils.save_as_image(base_fn + '_velocity.png', image, min_val=0, max_val=utils.max_depth) 


            # Visualize the results if needed
            if self.visualize_results:
                utils.save_as_image(os.path.join(base_fn, OTHER_DIR, img_num + '_snd_mono_2.png'), snd_mono_2, min_val=0, max_val=utils.max_depth) 
                utils.save_as_image(os.path.join(base_fn, OTHER_DIR, img_num + '_back.png'), back, min_val=0, max_val=utils.max_depth) 
                utils.save_as_image(os.path.join(base_fn, OTHER_DIR, img_num + '_fst_zeros.png'), fst_zeros, min_val=0, max_val=utils.max_depth) 
                #velocity[velocity > utils.max_velocity] = utils.max_velocity
                #velocity[velocity < -utils.max_velocity] = -utils.max_velocity
                
                # Save speed vectors as image
                #masked_speed = speed
                #masked_speed[~speed_mask] = 0
                
                utils.save_as_image(os.path.join(base_fn, MASK_DIR, img_num + '_speed_masked.png'), speed_mask*50, min_val=0, max_val=utils.max_depth) 
                return #! Doesn't work after this
                utils.save_as_image(base_fn + OTHER_DIR + img_num + '_speed.png', speed, min_val=0, max_val=utils.max_depth) 
                
                # Save velocity vectors as image
                for idx, file_id in enumerate(['x', 'y', 'z']):
                    utils.save_as_image('{}_{}.png'.format(base_fn + OTHER_DIR + img_num, file_id), 
                                        velocity[:, :, idx], max_val=utils.max_velocity)
                # Save depth vectors as image
                """for file_id, data in zip(['d1', 'd2'],
                                        [fst_depth, snd_depth]):
                    utils.save_as_image('{}_{}.png'.format(base_fn + OTHER_DIR + img_num, file_id), 
                                        data, min_val=0, max_val=utils.max_depth)"""
                # Save optical flow as image
                cv2.imwrite(base_fn + OTHER_DIR + img_num + '_flow.png', computeColor.computeImg(flow))
                cv2.imwrite(base_fn + OTHER_DIR + img_num + '_backflow.png', computeColor.computeImg(back_flow))
                utils.save_as_image(base_fn + OTHER_DIR + img_num + '_of_mask.png', incons_img, min_val=0, max_val=utils.max_depth)
                utils.save_as_image(base_fn + OTHER_DIR + img_num + '_fst_mono.png', fst_mono, min_val=0, max_val=utils.max_depth)
                utils.save_as_image(base_fn + OTHER_DIR + img_num + '_snd_mono.png', snd_mono, min_val=0, max_val=utils.max_depth)

                cv2.imwrite(base_fn + OTHER_DIR + img_num + '_flow_good_fst.png', computeColor.computeImg(good_flow))
                cv2.imwrite(base_fn + OTHER_DIR + img_num + '_flow_good_snd.png', computeColor.computeImg(good_flow_snd))
                # Save the final image
                visualize(base_fn + OTHER_DIR + img_num + '_viz.png', speed.astype('uint8'), velocity, base_fn + OTHER_DIR + img_num + '_z.png', 
                        fst_img, snd_img, base_fn + OTHER_DIR + img_num + '_fst_mono.png', base_fn + OTHER_DIR + img_num + '_snd_mono_2.png', 
                        base_fn + OTHER_DIR + img_num + '_flow_good_fst.png')

class GTVelocityCalculator(VelocityCalculator):
    def __init__(self, *args):
        super(GTVelocityCalculator, self).__init__(*args)
        
        self.read_depth = utils.read_gt_depth
        self.read_flow = utils.read_gt_flow
        self.average = utils.average_gt
        self.calculate_shifted_labels = lambda labels, avg_flow: labels 


def visualize(out_fn, labels, velocity, z_fn, fst_img, snd_img, 
            fst_depth_fn, snd_depth_fn, flow_fn):
    #img = utils.draw_velocity_vectors(labels, velocity, z_fn)

    fst_depth = cv2.imread(fst_depth_fn, cv2.IMREAD_COLOR)
    snd_depth = cv2.imread(snd_depth_fn, cv2.IMREAD_COLOR)
    flow = cv2.imread(flow_fn, cv2.IMREAD_COLOR)

    first_col = np.concatenate([fst_img, snd_img, img], axis=0)
    snd_col = np.concatenate([fst_depth, snd_depth, flow], axis=0)
    cv2.imwrite(out_fn, np.concatenate([first_col, snd_col], axis=1))


def calculate_velocity_and_orientation_wrapper(params):
    velocity_calculator = VelocityCalculator(*params)
    velocity_calculator.calculate_velocity_and_orientation()


def calculate_gt_velocity_and_orientation_wrapper(params):
    velocity_calculator = GTVelocityCalculator(*params)
    velocity_calculator.calculate_velocity_and_orientation()


def main(super_pixel_method, img_dir, disp_dir, flow_dir, back_flow_dir,label_dir, disp2_dir = None, out_dir = OUT_PATH, use_slic = False, test_number = 2, n_sps = 100, visualize = False, high = 1, low = 0):
    
    """video = Video("0001")

    cap = skvideo.io.FFmpegReader(video.video)
    frame_nr, h, w, _ = cap.getShape()

    video_images = []
    print('Read Video...')
    with tqdm.tqdm(total=frame_nr) as pbar:
        for i,frame in enumerate(cap.nextFrame()):
            video_images.append(frame)
            pbar.update()
            
    cap.close()

    Image.fromarray(video_images[0]).save(out_dir + "Test1.png")
    Image.fromarray(video_images[1]).save(out_dir + "Test2.png")
    """
    
    #fst_img_fns, snd_img_fns = out_dir + "Test1.png", out_dir + "Test2.png"

    img_fns = utils.list_directory(img_dir)
    fst_img_fns, snd_img_fns = img_fns, img_fns

    if disp2_dir is None:
        disp_fns = utils.list_directory(disp_dir, extension='.npy')
        fst_disp_fns, snd_disp_fns = disp_fns, disp_fns
        flow_fns = utils.list_directory(flow_dir, extension='.flo')
        back_flow = utils.list_directory(back_flow_dir, extension='.flo')
        calculate_velocity = calculate_velocity_and_orientation_wrapper
    else:
        assert os.path.isdir(disp2_dir)
        fst_disp_fns = utils.list_directory(disp_dir, extension='.png')
        snd_disp_fns = utils.list_directory(disp2_dir, extension='.png')
        flow_fns = utils.list_directory(flow_dir, extension='.png')
        back_flow = "N"
        calculate_velocity = calculate_gt_velocity_and_orientation_wrapper
    
    if use_slic:
        label_fns = utils.list_directory(label_dir, extension='.pgm')
    else:
        label_fns = utils.list_directory(label_dir)

    
    if len(label_fns) > len(flow_fns):
        label_fns = label_fns

    #print(len(flow_fns))
    #print(len(fst_img_fns))
    #print(len(snd_img_fns))
    #print(len(fst_disp_fns))
    #print(len(snd_disp_fns))
    #print(len(label_fns))
    #assert len(flow_fns) == len(fst_img_fns) 
    assert len(fst_img_fns) == len(snd_img_fns) 
    #assert len(flow_fns) == len(fst_disp_fns)
    assert len(fst_disp_fns) == len(snd_disp_fns) 
    if back_flow_dir != None:
        assert len(flow_fns) == len(back_flow)
    label_fns = fst_img_fns # So it doesn't quit too early
    """
    params = zip(fst_img_fns, snd_img_fns, disp_fns, label_fns, flow_fns, 
                itertools.repeat(out_dir), itertools.repeat(use_slic), 
                itertools.repeat(n_sps), itertools.repeat(visualize))

    with multiprocessing.Pool() as pool:
        with tqdm.tqdm(total=len(fst_img_fns)) as pbar:
            for _ in pool.imap_unordered(calculate_velocity, params):
                pbar.update()"""


    params = zip(fst_img_fns, snd_img_fns, fst_disp_fns, snd_disp_fns, label_fns, flow_fns, back_flow, 
                itertools.repeat(out_dir), itertools.repeat(use_slic), 
                itertools.repeat(n_sps), itertools.repeat(visualize),
                itertools.repeat(high), itertools.repeat(low), itertools.repeat(super_pixel_method))

    with multiprocessing.Pool() as pool:
        with tqdm.tqdm(total=len(fst_img_fns)) as pbar:
            for _ in pool.imap_unordered(calculate_velocity, params):
                pbar.update()


    

if __name__ == "__main__":
    run()
