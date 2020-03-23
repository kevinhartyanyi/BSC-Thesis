from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject
import time
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


def createSpeedErrorPlotMain(params):
    speeds, ground_truth, i, out_dir = params
    print("Creating speed and error plot {0}".format(i))

    est, = plt.plot(speeds[:i], "m", label="Estimated Speed")
    gt, = plt.plot(ground_truth[:i], "b", label="True Speed")
    plt.ylabel("Speed in km/h")
    plt.xlabel("Frame number")
    plt.legend(handles=[est,gt])
    #plt.xticks(np.arange(0, len(speeds), 1))
    plt.grid(axis='y', linestyle='-')
    plt.savefig(os.path.join(out_dir, "{0}_speed.png".format(i)), bbox_inches='tight', dpi=150) 

def createErrorPlotMain(params):
    error, i, out_dir = params

    print("Creating plot {0}".format(i))
    plt.plot(error[:i], "r")
    plt.ylabel("Error in km/h")
    plt.xlabel("Frame number")
    #plt.xticks(np.arange(0, len(speeds), 1))
    plt.grid(axis='y', linestyle='-')
    plt.savefig(os.path.join(out_dir, "{0}_error.png".format(i)), bbox_inches='tight', dpi=150)
    
def createSuperPixelMain(params):
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
    speeds, i, out_dir = params
    print("Creating speed plot {0}".format(i))

    plt.plot(speeds[:i], "m")
    plt.ylabel("Speed in km/h")
    plt.xlabel("Frame number")
    #plt.xticks(np.arange(0, len(speeds), 1))
    plt.grid(axis='y', linestyle='-')
    plt.savefig(os.path.join(out_dir, "{0}_speed.png".format(i)), bbox_inches='tight', dpi=150)
        

def ofMain(params):
    (img1, img2, ind), of_dir, of_model = params
    print("Running optical flow on:", img1, img2)
    flo_file = os.path.join(of_dir,"{0}.flo".format(ind))
    pwc.setupArguments(model=of_model, first_img=img1,
    second_img=img2, save_path=flo_file)
    pwc.run()
    # Transform from flo to png
    flow = fz.convert_from_file(flo_file)
    Image.fromarray(flow).save(os.path.join(of_dir,"{0}.png".format(ind)))

class CalculationRunner(QObject):
    finished = pyqtSignal()

    ofStart = pyqtSignal()
    depthStart = pyqtSignal()
    runStart = pyqtSignal()
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
        self.ground_truth_error = False
        self.video_frame = 0

    @pyqtSlot()
    def startCalc(self):
        img_fns = utils.list_directory(self.img_dir)
        fst_img_fns, snd_img_fns = img_fns, img_fns

        disp_fns = utils.list_directory(self.depth_dir, extension='.npy')
        fst_disp_fns, snd_disp_fns = disp_fns, disp_fns
        flow_fns = utils.list_directory(self.of_dir, extension='.flo')
        back_flow = utils.list_directory(self.back_of_dir, extension='.flo')
        calculate_velocity = calculate_velocity_and_orientation_wrapper


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
        if self.back_of_dir != None:
            assert len(flow_fns) == len(back_flow)
        #label_fns = fst_img_fns # So it doesn't quit too early
        label_fns_files = utils.list_directory(self.super_pixel_label_dir, extension=".npy")
        label_fns = []
        for label in label_fns_files:
            label_fns.append(np.load(label))

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
        img_list = utils.list_directory(self.img_dir)
        of_list = [(img_list[ind], img_list[ind+1], ind) for ind in range(len(img_list) - 1)]
        for ind in range(len(img_list) - 1):
            print("Running optical flow on:", img_list[ind], img_list[ind+1])
            flo_file = os.path.join(self.of_dir,"{0}.flo".format(ind))
            pwc.setupArguments(model=self.of_model, first_img=img_list[ind],
            second_img=img_list[ind+1], save_path=flo_file)
            pwc.run()
            # Transform from flo to png
            flow = fz.convert_from_file(flo_file)
            Image.fromarray(flow).save(os.path.join(self.of_dir,"{0}.png".format(ind)))
            self.update.emit(ind)

    @pyqtSlot()
    def startOf_new(self):
        img_list = utils.list_directory(self.img_dir)
        of_list = [(img_list[ind], img_list[ind+1], ind) for ind in range(len(img_list) - 1)]
        params = zip(of_list, itertools.repeat(self.of_dir), itertools.repeat(self.of_model)) 
        self.startMultiFunc(ofMain, params)

    @pyqtSlot()
    def startMultiFunc(self, run_func, params): # A slot takes no params
        #multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool() as pool:
            count = 1
            for _, i in enumerate(pool.imap_unordered(run_func, params)):
                self.update.emit(count)
                count += 1
    

    @pyqtSlot()
    def startBackOf(self): # A slot takes no params

        img_list = utils.list_directory(self.img_dir)
        for ind in reversed(range(len(img_list) - 1)):
            print("Running back optical flow on:", img_list[ind], img_list[ind-1])
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
        img_list = utils.list_directory(self.img_dir)
        for i, img in enumerate(img_list):
            print("Running depth estimation on:", img)
            monodepth.run(image_path=img, checkpoint_path=self.depth_model, save_path=os.path.join(self.depth_dir, ""))
            self.update.emit(i)
        


    @pyqtSlot()
    def startThread(self): # A slot takes no params        
        if self.send_video_frame:
            print("Only run image create")
            self.imagesFromVideo(self.vid_path, self.img_dir, "vid")
            self.videoFrame.emit(self.video_frame)
            return
        print("Start Main")
        self.checkRun("Super_Pixel_Label", self.createSuperPixel)
        #if self.super_pixel_method != "" and len(os.path.join(self.out_dir, "Super_Pixel", self.super_pixel_method) == 0):
        #    self.createSuperPixel()
        self.labelUpdate.emit(self.run_dict["Speed"])
        self.startCalc()
        
        self.checkRun("Error_Plot", self.createErrorPlot)
        self.checkRun("Super_Pixel_Video", self.createVid, os.path.join(self.out_dir, self.super_pixel_dir), self.out_dir, "super_pixel.mp4")
        print("End Main")
        self.finished.emit()
        return



        #self.checkRun("Video", self.imagesFromVideo, self.vid_path, self.img_dir, "vid")

        self.checkRun("Of", self.startOf)

        self.checkRun("Back_Of", self.startBackOf)
        self.checkRun("Depth", self.startDepth)
        #
        self.labelUpdate.emit(self.run_dict["Speed"])
        self.startCalc()

        self.updateFin.emit()
        #
        self.checkRun("Of_Vid", self.createVid, self.of_dir, self.out_dir, "of.mp4")
        self.checkRun("Back_Of_Vid", self.createVid, self.back_of_dir, self.out_dir, "back_of.mp4")
        self.checkRun("Depth_Vid", self.createVid, self.depth_dir, self.out_dir, "depth.mp4")
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
            
        self.checkRun("Speed_Plot_Video", self.createVid, os.path.join(self.out_dir, self.plot_speed_dir), self.out_dir, "speed_plot.mp4")
        if not self.ground_truth_error:
            self.checkRun("Error_Plot_Video", self.createVid, os.path.join(self.out_dir, self.plot_error_dir), self.out_dir, "error_plot.mp4")

        self.checkRun("Super_Pixel_Video", self.createVid, os.path.join(self.out_dir, self.super_pixel_dir), self.out_dir, "super_pixel.mp4")

        self.finished.emit()

    @pyqtSlot()
    def imagesFromVideo(self, path_to_video, save_path, vid_name):
        cam = cv2.VideoCapture(path_to_video) 
        
        # frame 
        currentframe = 0
        ret,frame = cam.read() 
        while(ret):
            name = os.path.join(save_path, "{0}_{1}.png".format(currentframe, vid_name))
            print ('Creating...' + name) 
    
            cv2.imwrite(name, frame) 
            currentframe += 1
            
            #self.update.emit(currentframe)
            ret,frame = cam.read() 

        
        self.video_frame = currentframe
        # Release all space and windows once done 
        cam.release() 
        cv2.destroyAllWindows() 
        

    def createSuperPixel(self):
        print("Create SUPER pixel label")
        images = utils.list_directory(self.img_dir, extension=".png")
        params = zip(itertools.repeat(self.super_pixel_method), itertools.repeat(images), range(0, len(images)), itertools.repeat(self.super_pixel_label_dir))
        self.startMultiFunc(createSuperPixelMain, params)

    def createSpeedPlot(self):
        plt.clf()
        speeds_dir = utils.list_directory(os.path.join(self.out_dir, self.numbers_dir), extension='speed.npy')
        speeds_dir = natsorted(speeds_dir)
        speeds = []
        for i, s in enumerate(speeds_dir):
            speeds.append(np.load(s))
        
        params = zip(itertools.repeat(speeds), range(1, len(speeds) + 1), itertools.repeat(os.path.join(self.out_dir, self.plot_speed_dir)))
        self.startMultiFunc(createSpeedPlotMain, params)

    @pyqtSlot()
    def createSpeedPlot_old(self):
        plt.clf()
        speeds_dir = utils.list_directory(os.path.join(self.out_dir, self.numbers_dir), extension='speed.npy')
        speeds_dir = natsorted(speeds_dir)
        speeds = []
        for i, s in enumerate(speeds_dir):
            speeds.append(np.load(s))
            print("Creating speed plot {0}".format(i))

            plt.plot(speeds, "m")
            plt.ylabel("Speed in km/h")
            plt.xlabel("Frame number")
            #plt.xticks(np.arange(0, len(speeds), 1))
            plt.grid(axis='y', linestyle='-')
            plt.savefig(os.path.join(os.path.join(self.out_dir, self.plot_speed_dir), "{0}_speed.png".format(i)), bbox_inches='tight', dpi=150)
            
            self.update.emit(i)


    def createErrorPlot(self):
        plt.clf()
        speeds_dir = utils.list_directory(os.path.join(self.out_dir, self.numbers_dir), extension='speed.npy')
        speeds_dir = natsorted(speeds_dir)
        speeds = []
        for i, s in enumerate(speeds_dir):
            speeds.append(np.load(s))
        
        csv = None
        if self.create_csv:
            csv = os.path.join(self.out_dir, '_error_Simple_OF.csv')
        try:
            _ = utils.error_comparison_Speed_Vecors(speeds,self.speed_gt[1:],csv=csv)
        except:
            self.error.emit("Error with the Ground Truth file, doesn't have correct shape")
            self.ground_truth_error = True
            return

        error = self.speed_gt[1:] - speeds


        params = zip(itertools.repeat(error), range(1, len(error) + 1), itertools.repeat(os.path.join(self.out_dir, self.plot_error_dir)))
        
        self.startMultiFunc(createErrorPlotMain, params)


    @pyqtSlot()
    def createErrorPlot_old(self):
        plt.clf()
        speeds_dir = utils.list_directory(os.path.join(self.out_dir, self.numbers_dir), extension='speed.npy')
        speeds_dir = natsorted(speeds_dir)
        speeds = []
        for s in speeds_dir:
            speeds.append(np.load(s))

        _ = utils.error_comparison_Speed_Vecors(speeds,self.speed_gt[1:],csv=os.path.join(self.out_dir, '_error_Simple_OF.csv'))

        error = self.speed_gt[1:] - speeds
        for i in range(1, len(error) + 1):
            print("Creating plot {0}".format(i))

            plt.plot(error[:i], "r")
            plt.ylabel("Error in km/h")
            plt.xlabel("Frame number")
            #plt.xticks(np.arange(0, len(speeds), 1))
            plt.grid(axis='y', linestyle='-')
            plt.savefig(os.path.join(os.path.join(self.out_dir, self.plot_error_dir), "{0}_error.png".format(i)), bbox_inches='tight', dpi=150)
            
            self.update.emit(i)


    def createSpeedErrorPlot(self):
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
    def createSpeedErrorPlot_old(self):
        plt.clf()
        speeds_dir = utils.list_directory(os.path.join(self.out_dir, self.numbers_dir), extension='speed.npy')
        speeds_dir = natsorted(speeds_dir)
        speeds = []
        for s in speeds_dir:
            speeds.append(np.load(s))

        ground_truth = self.speed_gt[1:]
        for i in range(1, len(ground_truth) + 1):
            print("Creating speed and error plot {0}".format(i))

            est, = plt.plot(speeds[:i], "m", label="Estimated Speed")
            gt, = plt.plot(ground_truth[:i], "b", label="True Speed")
            plt.ylabel("Speed in km/h")
            plt.xlabel("Frame number")
            plt.legend(handles=[est,gt])
            #plt.xticks(np.arange(0, len(speeds), 1))
            plt.grid(axis='y', linestyle='-')
            plt.savefig(os.path.join(os.path.join(self.out_dir, self.plot_speed_dir), "{0}_speed.png".format(i)), bbox_inches='tight', dpi=150)
            self.update.emit(i)

    @pyqtSlot()
    def checkRun(self, run_item, run_function, *args, **kwargs):
        if self.run_dict[run_item]["Run"]:
            self.labelUpdate.emit(self.run_dict[run_item])
            run_function(*args, **kwargs)
            self.updateFin.emit()

    def createVid(self, images_path, save_path, vid_name, fps=30):
        images = utils.list_directory(images_path, extension=".png")
        frame = cv2.imread(images[0])
        height, width, layers = frame.shape
        out = cv2.VideoWriter(os.path.join(save_path, vid_name),cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for i in range(len(images)):
            print("Writing frame: {0}".format(i))
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