from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject
import time
import speed.speed_vectors as speed
import speed.utils as utils
from speed.speed_vectors import calculate_velocity_and_orientation_wrapper
import itertools
import tqdm
import multiprocessing
import speed.pwc.run as pwc
import speed.monodepth.monodepth_simple as monodepth
import os
import cv2
import flowiz as fz
from PIL import Image


class CalculationRunner(QObject):
    finished = pyqtSignal()

    ofStart = pyqtSignal()
    depthStart = pyqtSignal()
    runStart = pyqtSignal()
    updateFin = pyqtSignal()
    labelUpdate = pyqtSignal(object)
    update = pyqtSignal(int)

    def __init__(self, img_dir, depth_dir, of_dir, back_of_dir, save_dir, label_dir, high, low, run_dict,
                of_model, depth_model):
        super(CalculationRunner, self).__init__()
        self.running = True
        self.use_slic = False
        self.visualize = True
        self.high = high
        self.low = low
        self.n_sps = 100
        self.run_dict = run_dict
        self.out_dir = save_dir
        self.label_dir = label_dir
        self.vid_name = "Video"
        self.of_model = of_model
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.depth_model = depth_model
        self.flow_dir = of_dir
        self.back_flow_dir = back_of_dir

    @pyqtSlot()
    def startCalc(self):
        img_fns = utils.list_directory(self.img_dir)
        fst_img_fns, snd_img_fns = img_fns, img_fns

        disp_fns = utils.list_directory(self.depth_dir, extension='.npy')
        fst_disp_fns, snd_disp_fns = disp_fns, disp_fns
        flow_fns = utils.list_directory(self.flow_dir, extension='.flo')
        back_flow = utils.list_directory(self.back_flow_dir, extension='.flo')
        calculate_velocity = calculate_velocity_and_orientation_wrapper

        if self.label_dir != None:
            label_fns = utils.list_directory(self.label_dir)

            
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
        if self.back_flow_dir != None:
            assert len(flow_fns) == len(back_flow)
        label_fns = fst_img_fns # So it doesn't quit too early
        
        

        params = zip(fst_img_fns, snd_img_fns, fst_disp_fns, snd_disp_fns, label_fns, flow_fns, back_flow, 
                    itertools.repeat(self.out_dir), itertools.repeat(self.use_slic), 
                    itertools.repeat(self.n_sps), itertools.repeat(self.visualize),
                    itertools.repeat(self.high), itertools.repeat(self.low), itertools.repeat(self.vid_name))

        with multiprocessing.Pool() as pool:
            with tqdm.tqdm(total=len(fst_img_fns)) as pbar:
                count = 1
                for _, i in enumerate(pool.imap_unordered(calculate_velocity, params)):
                    self.update.emit(count)
                    pbar.update()
                    count += 1
            

    @pyqtSlot()
    def startOf(self): # A slot takes no params

        img_list = utils.list_directory(self.img_dir)
        for ind in range(len(img_list) - 1):
            print("Running optical flow on:", img_list[ind], img_list[ind+1])
            flo_file = os.path.join(self.flow_dir,"{0}.flo".format(ind))
            pwc.setupArguments(model=self.of_model, first_img=img_list[ind],
            second_img=img_list[ind+1], save_path=flo_file)
            pwc.run()

            # Transform from flo to png
            flow = fz.convert_from_file(flo_file)
            Image.fromarray(flow).save(os.path.join(self.flow_dir,"{0}.png".format(ind)))
            self.update.emit(ind)

    @pyqtSlot()
    def startBackOf(self): # A slot takes no params

        img_list = utils.list_directory(self.img_dir)
        for ind in reversed(range(len(img_list) - 1)):
            print("Running back optical flow on:", img_list[ind], img_list[ind-1])
            back_flo_file = os.path.join(self.back_flow_dir,"{0}.flo".format(ind))
            pwc.setupArguments(model=self.of_model, first_img=img_list[ind],
            second_img=img_list[ind-1], save_path=back_flo_file)
            pwc.run()

            # Transform from flo to png
            flow = fz.convert_from_file(back_flo_file)
            Image.fromarray(flow).save(os.path.join(self.back_flow_dir,"{0}.png".format(ind)))
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
        #self.checkRun("Of", self.startOf)
        #self.checkRun("Back_Of", self.startBackOf)
        #self.checkRun("Depth", self.startDepth)
        #self.labelUpdate.emit(self.run_dict["Speed"])
        #self.startCalc()
        #self.updateFin.emit()
        self.createVid(images_path=self.img_dir, save_path=self.out_dir, vid_name="test.mp4")
        self.finished.emit()

    @pyqtSlot()
    def checkRun(self, run_item, run_function):
        if self.run_dict[run_item]["Run"]:
            self.labelUpdate.emit(self.run_dict[run_item])
            run_function()
            self.updateFin.emit()

    def createVid(self, images_path, save_path, vid_name, fps=30):
        images = utils.list_directory(images_path, extension=".png")

        frame = cv2.imread(images[0])
        height, width, layers = frame.shape
        out = cv2.VideoWriter(os.path.join(save_path, vid_name),cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
        for i in range(len(images)):
            print("Writing frame: {0}".format(i))
            # writing to a image array
            out.write(cv2.imread(images[i]))
            self.update.emit(i)
        out.release()

    def stop(self):
        """
        Stops the counting
        """
        self.running = False