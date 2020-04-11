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
from skimage.segmentation import mark_boundaries

from natsort import natsorted # Run with python3

import Model.Algorithms.speed.readFlowFile as readFlowFile
import Model.Algorithms.speed.computeColor as computeColor

import Model.Algorithms.utils as utils

import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt

result_dir = utils.getResultDirs()
RESULTS = result_dir["Results"]
OTHER_DIR = result_dir["Other"]
VL_DIR = result_dir["Velocity"]
NP_DIR = result_dir["Numbers"]
MASK_DIR = result_dir["Mask"]
DRAW_DIR = result_dir["Draw"]
SUPER_PIXEL_DIR = result_dir["SuperPixel"]
PLOT_CRASH_DIR = result_dir["Plot_Crash"]


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
        self.super_pixel_method = super_pixel_method
        self.create_draw = create_draw
        self.create_velocity = create_velocity


    def calculate_velocity_and_orientation(self):        
        if self.super_pixel_method != "":           

            # Read left and right images
            fst_img = cv2.imread(self.fst_img_fn, cv2.IMREAD_COLOR)  # height x width x channels
            snd_img = cv2.imread(self.snd_img_fn, cv2.IMREAD_COLOR)  # height x width x channels

            # Read optical flow
            flow = self.read_flow(self.flow_fn)
            avg_flow = np.zeros_like(flow, dtype=np.float32) 

            # Save the results
            base_fn = self.out_dir
            img_num = os.path.splitext(os.path.basename(self.flow_fn))[0]

            labels = self.label_fn
            labels_unique = np.unique(labels)
            label_masks = []
            for uni in labels_unique:
                label_masks.append(labels == uni)
            

            assert fst_img.shape == snd_img.shape
            height, width, _ = fst_img.shape
            # Read disparity maps
            fst_depth = self.read_depth(self.fst_depth_fn, width, height)
            avg_fst_depth = self.average(fst_depth, labels, labels_unique, label_masks)
            snd_depth = self.read_depth(self.snd_depth_fn, width, height)
            
            avg_flow[:, :, 0] = self.average(flow[:, :, 0], labels, labels_unique, label_masks)
            avg_flow[:, :, 1] = self.average(flow[:, :, 1], labels, labels_unique, label_masks)

            # Shift labels and depth values respect to the average optical flow
            shifted_labels = self.calculate_shifted_labels(labels, avg_flow)
            avg_shifted_depth = self.average(snd_depth, shifted_labels, labels_unique, label_masks)

            # Calculate the velocity and the orientation
            velocity = \
                utils.calculate_velocity_and_orientation_vectors(labels, shifted_labels, 
                                                                avg_flow, 
                                                                avg_fst_depth, 
                                                                avg_shifted_depth)

            speed, speed_mask = utils.vector_speedOF_Simple(velocity, low=self.low, high=self.high) # Speed calculation
            np.save(os.path.join(base_fn, NP_DIR, img_num + '_speed.npy'), speed)
            np.save(os.path.join(base_fn, NP_DIR, img_num + '_mask.npy'), speed_mask)
            utils.save_as_image(os.path.join(base_fn, MASK_DIR, img_num + '_speed_masked.png'), speed_mask*50, min_val=0, max_val=utils.max_depth) 
            np.save(os.path.join(base_fn, NP_DIR, img_num + '_velocity.npy'), velocity)

            if self.create_draw:
                back_flow = self.read_flow(self.back_flow) 
                of_mask, next_position, prev_position = utils.calc_bidi_errormap(flow, back_flow, tau=0.8)
                
                incons_img = np.tile(255*np.ones(of_mask[..., None].shape), (1, 1, 3)).astype(np.uint8)
                incons_img = utils.draw_velocity_vectors(incons_img, next_position, relative_disp=False, color=(0, 0, 255))
                plt.imsave(os.path.join(base_fn, DRAW_DIR, img_num + '_draw.png'), incons_img.astype('uint8'))


            if self.create_velocity:
                image = velocity.copy()
                plt.imsave(os.path.join(base_fn, VL_DIR, img_num + '_velocity.png'), image.astype('uint8'))


            x = velocity[:,:,0]
            y = velocity[:,:,1]
            z = velocity[:,:,2]
            speed_superpixel = utils.vector_distance(x,y,z)
            #mask_edges = mark_boundaries(speed_superpixel.astype(np.int), speed_mask, color=(1,0,0))

            np.save(os.path.join(base_fn, NP_DIR, img_num + '_superpixel.npy'), speed_superpixel)
            plt.matshow(speed_superpixel, vmin=0, vmax=100)
            plt.colorbar()
            plt.savefig(os.path.join(base_fn, SUPER_PIXEL_DIR, "{0}_superpixel.png".format(img_num)), bbox_inches='tight', dpi=150)
        else:
            # Read left and right images
            fst_img = cv2.imread(self.fst_img_fn, cv2.IMREAD_COLOR)  # height x width x channels
            snd_img = cv2.imread(self.snd_img_fn, cv2.IMREAD_COLOR)  # height x width x channels

            assert fst_img.shape == snd_img.shape
            height, width, _ = fst_img.shape
            
            # Read disparity maps
            fst_depth = self.read_depth(self.fst_depth_fn, width, height)
            snd_depth = self.read_depth(self.snd_depth_fn, width, height)
            # Read optical flow
            flow = self.read_flow(self.flow_fn)
            back_flow = self.read_flow(self.back_flow)            

            of_mask, next_position, prev_position = utils.calc_bidi_errormap(flow, back_flow, tau=0.8)
            
            incons_img = np.tile(255*np.ones(of_mask[..., None].shape), (1, 1, 3)).astype(np.uint8)
            incons_img = utils.draw_velocity_vectors(incons_img, next_position, relative_disp=False, color=(0, 0, 255))
            
            
            
            # Save the results
            base_fn = self.out_dir
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

            
            if self.create_draw:
                plt.imsave(os.path.join(base_fn, DRAW_DIR, img_num + '_draw.png'), incons_img.astype('uint8'))

            speed, speed_mask = utils.vector_speedOF_Simple(velocity,low=self.low,high=self.high) # Speed calculation
            np.save(os.path.join(base_fn, NP_DIR, img_num + '_speed.npy'), speed)
            np.save(os.path.join(base_fn, NP_DIR, img_num + '_mask.npy'), speed_mask)
            np.save(os.path.join(base_fn, NP_DIR, img_num + '_velocity.npy'), velocity)

            utils.save_as_image(os.path.join(base_fn, MASK_DIR, img_num + '_speed_masked.png'), speed_mask*50, min_val=0, max_val=utils.max_depth) 

def calculate_velocity_and_orientation_wrapper(params):
    velocity_calculator = VelocityCalculator(*params)
    velocity_calculator.calculate_velocity_and_orientation()
