from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject
import time
import speed.speed_vectors as speed
import speed.utils as utils
from speed.speed_vectors import calculate_velocity_and_orientation_wrapper
import itertools
import tqdm
import multiprocessing
import speed.pwc.run as pwc
import os
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
                of_model):
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
        self.disp_dir = depth_dir
        self.flow_dir = of_dir
        self.back_flow_dir = back_of_dir

    @pyqtSlot()
    def startCalc(self):
        img_fns = utils.list_directory(self.img_dir)
        fst_img_fns, snd_img_fns = img_fns, img_fns

        disp_fns = utils.list_directory(self.disp_dir, extension='.npy')
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
    def startDepth(self): # A slot takes no params
        time.sleep(1)
        print("Emit")
        self.update.emit(1)
        time.sleep(1)
        print("Emit")
        self.update.emit(2)
        time.sleep(1)
        print("Emit")
        self.update.emit(3)
    
    @pyqtSlot()
    def startThread(self): # A slot takes no params
        self.startOf()
        ##if self.run_dict["Of"]["Run"]:
        #    self.labelUpdate.emit(self.run_dict["Of"])
        #    self.startOf()
        #    self.updateFin.emit()
        ##if self.run_dict["Depth"]["Run"]:
        #    self.labelUpdate.emit(self.run_dict["Depth"])
        #    self.startDepth()
        #    self.updateFin.emit()
        #self.labelUpdate.emit(self.run_dict["Speed"])
        #self.startCalc()
        #self.updateFin.emit()

        #self.finished.emit()

    def stop(self):
        """
        Stops the counting
        """
        self.running = False