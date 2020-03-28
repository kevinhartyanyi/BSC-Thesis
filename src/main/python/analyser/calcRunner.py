from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject
import speed.speed_vectors as speed
import speed.utils as utils
from speed.speed_vectors import calculate_velocity_and_orientation_wrapper
import itertools
import tqdm
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.color import rgb2gray
from skimage.filters import sobel
import multiprocessing
import speed.pwc.run as pwc
import speed.monodepth.monodepth_simple as monodepth
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import flowiz as fz
from PIL import Image
import logging
from scipy.optimize import minimize

def createSpeedErrorPlotMain(params):
    """Creates combined plot with speed and error values
    
    Arguments:
        params {tuple} -- parameters for the function
    """
    speeds, ground_truth, i, out_dir = params
    logging.info("Creating speed and error plot {0}".format(i))

    est, = plt.plot(speeds[:i], "m", label="Estimated Speed")
    gt, = plt.plot(ground_truth[:i], "b", label="True Speed")
    plt.ylabel("Speed in km/h")
    plt.xlabel("Frame number")
    plt.legend(handles=[est,gt])
    #plt.xticks(np.arange(0, len(speeds), 1))
    plt.grid(axis='y', linestyle='-')
    plt.savefig(os.path.join(out_dir, "{0}_speed.png".format(i)), bbox_inches='tight', dpi=150) 

def createErrorPlotMain(params):
    """Creates error plot
    
    Arguments:
        params {tuple} -- parameters for the function
    """
    error, i, out_dir = params

    logging.info("Creating plot {0}".format(i))
    plt.plot(error[:i], "r")
    plt.ylabel("Error in km/h")
    plt.xlabel("Frame number")
    #plt.xticks(np.arange(0, len(speeds), 1))
    plt.grid(axis='y', linestyle='-')
    plt.savefig(os.path.join(out_dir, "{0}_error.png".format(i)), bbox_inches='tight', dpi=150)
    
def createSuperPixelMain(params):
    """Calculates super pixel segmentation
    
    Arguments:
        params {tuple} -- parameters for the function
    """
    super_pixel_method, img_file, i, out_dir = params

    img = cv2.imread(img_file[i], cv2.IMREAD_COLOR)

    if super_pixel_method == "Felzenszwalb":
        labels = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)                
    elif super_pixel_method == "Quickshift":
        labels = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    elif super_pixel_method == "Slic":
        labels = slic(img, n_segments=250, compactness=10, sigma=1)
    elif super_pixel_method == "Watershed":
        gradient = sobel(rgb2gray(img))
        labels = watershed(gradient, markers=250, compactness=0.001)
    
    np.save(os.path.join(out_dir, "{0}_{1}.npy".format(i, super_pixel_method)), labels)

def createSpeedPlotMain(params):
    """Creates speed plot
    
    Arguments:
        params {tuple} -- parameters for the function
    """
    speeds, i, out_dir = params
    logging.info("Creating speed plot {0}".format(i))

    plt.plot(speeds[:i], "m")
    plt.ylabel("Speed in km/h")
    plt.xlabel("Frame number")
    #plt.xticks(np.arange(0, len(speeds), 1))
    plt.grid(axis='y', linestyle='-')
    plt.savefig(os.path.join(out_dir, "{0}_speed.png".format(i)), bbox_inches='tight', dpi=150)
        

class CalculationRunner(QObject):
    finished = pyqtSignal()

    #ofStart = pyqtSignal()
    #depthStart = pyqtSignal()
    #runStart = pyqtSignal()
    updateFin = pyqtSignal()
    labelUpdate = pyqtSignal(object)
    update = pyqtSignal(int)
    videoFrame = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, param_dict):
        super(CalculationRunner, self).__init__()
        self.running = True
        self.use_slic = False
        self.visualize = True
        self.vid_path = param_dict["vid_path"]
        self.high = param_dict["high"]
        self.speed_gt = np.load(param_dict["speed_gt"]) if param_dict["speed_gt"] != "" else "" 
        self.low = param_dict["low"]
        self.n_sps = 100
        self.run_dict = param_dict["run_dict"]
        self.out_dir = param_dict["save_dir"]
        self.plot_error_dir = param_dict["plot_error_dir"]
        self.numbers_dir = param_dict["numbers_dir"]
        self.super_pixel_method = param_dict["super_pixel_method"]
        self.super_pixel_dir = param_dict["super_pixel_dir"]
        self.of_model = param_dict["of_model"]
        self.img_dir = param_dict["img_dir"]
        self.depth_dir = param_dict["depth_dir"]
        self.depth_model = param_dict["depth_model"]
        self.of_dir = param_dict["of_dir"]
        self.back_of_dir = param_dict["back_of_dir"]
        self.plot_speed_dir = param_dict["plot_speed_dir"]
        self.send_video_frame = param_dict["send_video_frame"]
        self.create_csv = param_dict["create_csv"]
        self.create_draw = param_dict["create_draw"]
        self.create_velocity = param_dict["create_velocity"]
        self.super_pixel_label_dir = param_dict["super_pixel_label_dir"]
        self.create_video_fps = param_dict["create_video_fps"]
        self.optimize = param_dict["optimize_params"]
        self.ground_truth_error = False
        self.video_frame = 0



    @pyqtSlot()
    def startThread(self): # A slot takes no params  
        """Starts the calculations
        """   
        if self.send_video_frame:
            logging.info("Only run image create")
            self.imagesFromVideo(self.vid_path, self.img_dir, "vid")
            self.videoFrame.emit(self.video_frame)
            return
        logging.info("Start Main Run")
        #if self.super_pixel_method != "" and len(os.path.join(self.out_dir, "Super_Pixel", self.super_pixel_method) == 0):
        #    self.createSuperPixel()
        
        


        #self.checkRun("Video", self.imagesFromVideo, self.vid_path, self.img_dir, "vid")

        self.checkRun("Of", self.startOf)

        self.checkRun("Back_Of", self.startBackOf)
        self.checkRun("Depth", self.startDepth)
        self.checkRun("Super_Pixel_Label", self.createSuperPixel)
        #
        if self.optimize:
            logging.info("Start Paramter Optimization")
            self.labelUpdate.emit(self.run_dict["Optimization"])
            self.startOptimization()
            self.updateFin.emit()
        else:
            logging.info("Start Calculation")
            self.labelUpdate.emit(self.run_dict["Speed"])
            self.startCalc()
            self.updateFin.emit()
        #
        self.checkRun("Of_Vid", self.createVid, self.of_dir, self.out_dir, "of.mp4", self.create_video_fps)
        self.checkRun("Back_Of_Vid", self.createVid, self.back_of_dir, self.out_dir, "back_of.mp4", self.create_video_fps)
        self.checkRun("Depth_Vid", self.createVid, self.depth_dir, self.out_dir, "depth.mp4", self.create_video_fps)
        #
        if self.run_dict["Error_Plot"]["Run"] and self.run_dict["Speed_Plot"]["Run"]:
            self.labelUpdate.emit(self.run_dict["Speed_Plot"])
            try:
                self.createSpeedErrorPlot()
                self.checkRun("Error_Plot", self.createErrorPlot)
            except:
                self.error.emit("Error with the Ground Truth file, doesn't have correct shape")
                self.ground_truth_error = True
                self.createSpeedPlot()
            self.updateFin.emit()
        else:
            self.checkRun("Error_Plot", self.createErrorPlot)
            self.checkRun("Speed_Plot", self.createSpeedPlot)
            
        self.checkRun("Speed_Plot_Video", self.createVid, os.path.join(self.out_dir, self.plot_speed_dir), self.out_dir, "speed_plot.mp4", self.create_video_fps)
        if not self.ground_truth_error:
            self.checkRun("Error_Plot_Video", self.createVid, os.path.join(self.out_dir, self.plot_error_dir), self.out_dir, "error_plot.mp4")

        self.checkRun("Super_Pixel_Video", self.createVid, os.path.join(self.out_dir, self.super_pixel_dir), self.out_dir, "super_pixel.mp4", self.create_video_fps)
        

        self.finished.emit()

    @pyqtSlot()
    def startCalc(self):
        """Start speed calculation with or without superpixels
        """
        img_fns = utils.list_directory(self.img_dir)
        fst_img_fns, snd_img_fns = img_fns, img_fns

        disp_fns = utils.list_directory(self.depth_dir, extension='.npy')
        fst_disp_fns, snd_disp_fns = disp_fns, disp_fns
        flow_fns = utils.list_directory(self.of_dir, extension='.flo')
        back_flow = utils.list_directory(self.back_of_dir, extension='.flo')
        calculate_velocity = calculate_velocity_and_orientation_wrapper


        assert len(fst_img_fns) == len(snd_img_fns) 
        assert len(fst_disp_fns) == len(snd_disp_fns) 
        if self.back_of_dir != None:
            assert len(flow_fns) == len(back_flow)
        #label_fns = fst_img_fns # So it doesn't quit too early
        if self.super_pixel_method != "":
            label_fns_files = utils.list_directory(self.super_pixel_label_dir, extension=".npy")
            label_fns = []
            for label in label_fns_files:
                label_fns.append(np.load(label))
        else:
            label_fns = fst_img_fns # So it doesn't quit too early

        params = zip(fst_img_fns, snd_img_fns, fst_disp_fns, snd_disp_fns, label_fns, flow_fns, back_flow, 
                    itertools.repeat(self.out_dir), itertools.repeat(self.use_slic), 
                    itertools.repeat(self.n_sps), itertools.repeat(self.visualize),
                    itertools.repeat(self.high), itertools.repeat(self.low), itertools.repeat(self.super_pixel_method), 
                    itertools.repeat(self.create_draw), itertools.repeat(self.create_velocity))

        with multiprocessing.Pool() as pool:
            with tqdm.tqdm(total=len(fst_img_fns)) as pbar:
                count = 1
                for _, i in enumerate(pool.imap_unordered(calculate_velocity, params)):
                    self.update.emit(count)
                    pbar.update()
                    count += 1
        
    @pyqtSlot()
    def startOf(self):
        """Start optical flow calculation
        """
        img_list = utils.list_directory(self.img_dir)
        of_list = [(img_list[ind], img_list[ind+1], ind) for ind in range(len(img_list) - 1)]
        for ind in range(len(img_list) - 1):
            logging.info("Running optical flow on: {0} {1}".format(img_list[ind], img_list[ind+1]))
            flo_file = os.path.join(self.of_dir,"{0}.flo".format(ind))
            pwc.setupArguments(model=self.of_model, first_img=img_list[ind],
            second_img=img_list[ind+1], save_path=flo_file)
            pwc.run()
            # Transform from flo to png
            flow = fz.convert_from_file(flo_file)
            Image.fromarray(flow).save(os.path.join(self.of_dir,"{0}.png".format(ind)))
            self.update.emit(ind)

    #@pyqtSlot()
    #def startOf_new(self):
    #    img_list = utils.list_directory(self.img_dir)
    #    of_list = [(img_list[ind], img_list[ind+1], ind) for ind in range(len(img_list) - 1)]
    #    params = zip(of_list, itertools.repeat(self.of_dir), itertools.repeat(self.of_model)) 
    #    self.startMultiFunc(ofMain, params)

    def con(params):
        if params[1] > params[0]:
            return 0
        else:
            return 1

    def minFunc(self, params):
        self.low = params[0]
        self.high = params[1]
        self.startCalc()

    @pyqtSlot()
    def startOptimization(self):
        """Parameter optimization
        """
        #cons = {'type':'eq', 'fun': con}
        #m = minimize(minFunc, [self.low, self.high], bounds=[(0.0,1.0), (0.0,1.0)], constraints=cons)
        #print(m)
        np_dir = utils.getResultDirs()["Numbers"]
        self.startCalc()
        speeds_dir = utils.list_directory(np_dir, extension='speed.npy')
        speeds = []
        for s in speeds_dir:
            speeds.append(np.load(s))
        rmse = utils.error_comparison_Speed_Vecors(speeds,self.speed_gt[1:])
        #errors.append(rmse)
        for s in speeds_dir:
            os.remove(s)
        print(rmse)
        input("Wait")

    @pyqtSlot()
    def startMultiFunc(self, run_func, params): # A slot takes no params
        """Runs the given funcion with the given parameters on multiple cpu cores
        Arguments:
            run_func {function} -- the function to run
            params {tuple} -- parameters for the function
        """
        #multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool() as pool:
            count = 1
            for _, i in enumerate(pool.imap_unordered(run_func, params)):
                self.update.emit(count)
                count += 1
    
    @pyqtSlot()
    def startBackOf(self): # A slot takes no params
        """Start backward optical flow calculation
        """

        img_list = utils.list_directory(self.img_dir)
        for ind in reversed(range(len(img_list) - 1)):
            logging.info("Running back optical flow on: {0} {1}".format(img_list[ind], img_list[ind-1]))
            back_flo_file = os.path.join(self.back_of_dir,"{0}.flo".format(ind))
            pwc.setupArguments(model=self.of_model, first_img=img_list[ind],
            second_img=img_list[ind-1], save_path=back_flo_file)
            pwc.run()

            # Transform from flo to png
            flow = fz.convert_from_file(back_flo_file)
            Image.fromarray(flow).save(os.path.join(self.back_of_dir,"{0}.png".format(ind)))
            self.update.emit(abs(ind - len(img_list)))
        
    @pyqtSlot()
    def startDepth(self): # A slot takes no params
        """Start depth estimation
        """
        img_list = utils.list_directory(self.img_dir)
        for i, img in enumerate(img_list):
            logging.info("Running depth estimation on: {0}".format(img))
            monodepth.run(image_path=img, checkpoint_path=self.depth_model, save_path=os.path.join(self.depth_dir, ""))
            self.update.emit(i)
        
    @pyqtSlot()
    def imagesFromVideo(self, path_to_video, save_path, vid_name):
        """Creates images from a video and saves them
        
        Arguments:
            path_to_video {str} -- path to the video
            save_path {str} -- path where the images will be saved
            vid_name {str} -- name of the video
        """
        cam = cv2.VideoCapture(path_to_video) 
        
        # frame 
        currentframe = 0
        ret,frame = cam.read() 
        while(ret):
            name = os.path.join(save_path, "{0}_{1}.png".format(currentframe, vid_name))
    
            cv2.imwrite(name, frame) 
            currentframe += 1
            
            #self.update.emit(currentframe)
            ret,frame = cam.read() 

        
        self.video_frame = currentframe
        # Release all space and windows once done 
        cam.release() 
        cv2.destroyAllWindows() 
        
    def createSuperPixel(self):
        """Start super pixel calculation on multiple cpu cores
        """
        logging.info("Create SUPER pixel label")
        images = utils.list_directory(self.img_dir, extension=".png")
        params = zip(itertools.repeat(self.super_pixel_method), itertools.repeat(images), range(0, len(images)), itertools.repeat(self.super_pixel_label_dir))
        self.startMultiFunc(createSuperPixelMain, params)

    def createSpeedPlot(self):
        """Start speed plot creation on multiple cpu cores
        """
        plt.clf()
        speeds_dir = utils.list_directory(os.path.join(self.out_dir, self.numbers_dir), extension='speed.npy')
        speeds_dir = natsorted(speeds_dir)
        speeds = []
        for i, s in enumerate(speeds_dir):
            speeds.append(np.load(s))
        
        params = zip(itertools.repeat(speeds), range(1, len(speeds) + 1), itertools.repeat(os.path.join(self.out_dir, self.plot_speed_dir)))
        self.startMultiFunc(createSpeedPlotMain, params)




    def createErrorPlot(self):
        """Start error plot creation on multiple cpu cores
        Sends signal with error message if ground truth has wrong shape
        """
        plt.clf()
        speeds_dir = utils.list_directory(os.path.join(self.out_dir, self.numbers_dir), extension='speed.npy')
        speeds_dir = natsorted(speeds_dir)
        speeds = []
        for i, s in enumerate(speeds_dir):
            speeds.append(np.load(s))
        
        csv = None
        if self.create_csv:
            csv = os.path.join(self.out_dir, "_error_Simple_OF.csv")
        try:
            _ = utils.error_comparison_Speed_Vecors(speeds,self.speed_gt[1:],csv=csv)
        except:
            self.error.emit("Error with the Ground Truth file, doesn't have correct shape")
            self.ground_truth_error = True
            return

        error = abs(self.speed_gt[1:] - speeds)


        params = zip(itertools.repeat(error), range(1, len(error) + 1), itertools.repeat(os.path.join(self.out_dir, self.plot_error_dir)))
        
        self.startMultiFunc(createErrorPlotMain, params)



#

    def createSpeedErrorPlot(self):
        """Start speed and ground truth speed comparison plot creation on multiple cpu cores
        Sends signal with error message if ground truth has wrong shape
        """
        plt.clf()
        speeds_dir = utils.list_directory(os.path.join(self.out_dir, self.numbers_dir), extension='speed.npy')
        speeds_dir = natsorted(speeds_dir)
        speeds = []
        for s in speeds_dir:
            speeds.append(np.load(s))

        ground_truth = self.speed_gt[1:]

        params = zip(itertools.repeat(speeds), itertools.repeat(ground_truth), range(1, len(ground_truth) + 1), itertools.repeat(os.path.join(self.out_dir, self.plot_speed_dir)))
        self.startMultiFunc(createSpeedErrorPlotMain, params)
        


    @pyqtSlot()
    def checkRun(self, run_item, run_function, *args, **kwargs):
        """Check if the given function should be run, and run it if yes
        
        Arguments:
            run_item {str} -- the name of the calculation in run_dict
            run_function {function} -- function that executes the calculation
        """
        if self.run_dict[run_item]["Run"]:
            self.labelUpdate.emit(self.run_dict[run_item])
            run_function(*args, **kwargs)
            self.updateFin.emit()

    def createVid(self, images_path, save_path, vid_name, fps=30):
        """Create video from the images in the given path with the given fps
        
        Arguments:
            images_path {str} -- path to images
            save_path {str} -- path to save the video
            vid_name {str} -- the name of the created video
        
        Keyword Arguments:
            fps {int} -- fps for the video (default: {30})
        """
        images = utils.list_directory(images_path, extension=".png")
        frame = cv2.imread(images[0])
        height, width, layers = frame.shape
        out = cv2.VideoWriter(os.path.join(save_path, vid_name),cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for i in range(len(images)):
            logging.info("Writing frame: {0}".format(i))
            # writing to a image array
            img = cv2.imread(images[i])
            resized = cv2.resize(img,(width,height)) 
            out.write(resized)
            self.update.emit(i)
        out.release()

    def stop(self):
        """
        Stops the counting
        """
        self.running = False