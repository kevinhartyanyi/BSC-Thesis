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

RESULTS = 'results'
OTHER_DIR = os.path.join(RESULTS, 'other')
VL_DIR = os.path.join(RESULTS, 'velocity')
NP_DIR = os.path.join(RESULTS, 'numbers')
MASK_DIR = os.path.join(RESULTS, 'mask')
DRAW_DIR = os.path.join(RESULTS, 'draw')
SUPER_PIXEL_DIR = os.path.join(RESULTS, 'super_pixel')



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

class VelocityCalculator(object):
    def __init__(self,fst_img_fn, snd_img_fn, fst_depth_fn, snd_depth_fn, 
                label_fn, flow_fn, back_flow, out_dir, use_slic, n_sps, visualize_results=True, high=1, low=0,
                super_pixel_method="", create_draw=False, create_velocity=False):
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
        self.create_draw = create_draw
        self.create_velocity = create_velocity


    def calculate_velocity_and_orientation(self):        
        if self.super_pixel_method != "":
            # Read superpixel labels
            #if self.use_slic:
            #    labels = utils.read_gslic_labels(self.label_fn, n_sps=self.n_sps)
            #else:
            #    labels = utils.read_boruvka_labels(self.label_fn, n_sps=self.n_sps)
            

            # Read left and right images
            fst_img = cv2.imread(self.fst_img_fn, cv2.IMREAD_COLOR)  # height x width x channels
            snd_img = cv2.imread(self.snd_img_fn, cv2.IMREAD_COLOR)  # height x width x channels

            #if self.super_pixel_method == "Felzenszwalb":
            #    labels = felzenszwalb(fst_img, scale=100, sigma=0.5, min_size=50)                
            #elif self.super_pixel_method == "Quickshift":
            #    labels = quickshift(fst_img, kernel_size=3, max_dist=6, ratio=0.5)
            #elif self.super_pixel_method == "Slic":
            #    labels = slic(fst_img, n_segments=250, compactness=10, sigma=1)
            #elif self.super_pixel_method == "Watershed":
            #    gradient = sobel(rgb2gray(fst_img))
            #    labels = watershed(gradient, markers=250, compactness=0.001)

            # Read optical flow
            flow = self.read_flow(self.flow_fn)
            avg_flow = np.zeros_like(flow) 

            # Save the results
            base_fn = self.out_dir
            img_num = os.path.splitext(os.path.basename(self.flow_fn))[0]

            #labels = cv2.imread(self.label_fn, -1)
            #labels = np.load(os.path.join(self.label_fn, "{0}_{1}.npy".format(img_num, self.super_pixel_method)))
            #labels = np.load(self.label_fn)
            labels = self.label_fn

            assert fst_img.shape == snd_img.shape
            height, width, _ = fst_img.shape
            # Read disparity maps
            fst_depth = self.read_depth(self.fst_depth_fn, width, height)
            avg_fst_depth = self.average(fst_depth, labels)
            snd_depth = self.read_depth(self.snd_depth_fn, width, height)
            
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

            

            speed, speed_mask = utils.vector_speedOF_Simple(velocity, low=self.low, high=self.high) # Speed calculation
            np.save(os.path.join(base_fn, NP_DIR, img_num + '_speed.npy'), speed)
            np.save(os.path.join(base_fn, NP_DIR, img_num + '_mask.npy'), speed_mask)
            utils.save_as_image(os.path.join(base_fn, MASK_DIR, img_num + '_speed_masked.png'), speed_mask*50, min_val=0, max_val=utils.max_depth) 

            if self.create_draw:
                back_flow = self.read_flow(self.back_flow) 
                of_mask, next_position, prev_position = utils.calc_bidi_errormap(flow, back_flow, tau=0.8)
                incons_img = np.tile(255*of_mask[..., None], (1, 1, 3)).astype(np.uint8)
                incons_img = utils.draw_velocity_vectors(incons_img, next_position, relative_disp=False, color=(0, 0, 255))
            
            if self.create_velocity:
                image = velocity.copy()
                plt.imsave(os.path.join(base_fn, VL_DIR, img_num + '_velocity.png'), image.astype('uint8'))

            x = velocity[:,:,0]
            y = velocity[:,:,1]
            z = velocity[:,:,2]
            speed_superpixel = utils.vector_distance(x,y,z)
            np.save(os.path.join(base_fn, NP_DIR, img_num + '_superpixel.npy'), speed_superpixel)
            plt.matshow(speed_superpixel)
            plt.colorbar()
            plt.savefig(os.path.join(base_fn, SUPER_PIXEL_DIR, "{0}_superpixel.png".format(img_num)), bbox_inches='tight', dpi=150)

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
            
            if self.create_velocity:
                image = velocity.copy()
                plt.imsave(os.path.join(base_fn, VL_DIR, img_num + '_velocity.png'), image.astype('uint8'))


            #inaccurate_mono = fst_depth > 115
            inaccurate_mono = fst_depth > 30
            fst_depth[inaccurate_mono] = 0
            velocity[inaccurate_mono] = 0

            
            #np.save(base_fn + '_vel.npy', velocity)
            #np.save(base_fn + '_ori.npy', orientation)
            if self.create_draw:
                plt.imsave(os.path.join(base_fn, DRAW_DIR, img_num + '_draw.png'), incons_img.astype('uint8'))

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

def calculate_velocity_and_orientation_wrapper(params):
    velocity_calculator = VelocityCalculator(*params)
    velocity_calculator.calculate_velocity_and_orientation()
