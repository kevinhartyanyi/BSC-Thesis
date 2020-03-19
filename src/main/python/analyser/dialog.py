from PyQt5.QtWidgets import QDialog, QFileDialog, QColorDialog, QProgressBar, QLabel, QMessageBox
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread
from dialogUI import Ui_Dialog
import platform
import json
import calcRunner
import os
import shutil
import speed.speed_vectors as speed
from speed.utils import list_directory
import re
import cv2

import speed.pwc.run as pwc


RESULTS = 'results'
OTHER_DIR = os.path.join(RESULTS, 'other')
VL_DIR = os.path.join(RESULTS, 'velocity')
NP_DIR = os.path.join(RESULTS, 'numbers')
MASK_DIR = os.path.join(RESULTS, 'mask')
DRAW_DIR = os.path.join(RESULTS, 'draw')
SUPER_PIXEL_DIR = os.path.join(RESULTS, 'super_pixel')
PLOT_SPEED_DIR = os.path.join(RESULTS, 'plot_speed')
PLOT_ERROR_DIR = os.path.join(RESULTS, 'plot_error')


class Dialog(QDialog, Ui_Dialog):
    sendUser = pyqtSignal(object)
    sendCreated = pyqtSignal(object)

    def __init__(self, app, parent=None):
        super(Dialog, self).__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle("Process Video")
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.user_file = os.path.join(self.dir, ".userInfo.json") 
        self.created = {}
        self.user = None
        self.thread = None
        self.progressBar = None
        self.progressAllBar = None
        self.progressLabel = None
        self.img_exist = False
        self.params_dict = None
        self.of_exist = False
        self.back_of_exist = False
        self.depth_exist = False
        self.vid_name = None
        self.all_run = 1
        self.run_count = 1
        self.fps = 30
        self.fps_limit = 60
        self.run_dict = {}
        self.super_pixel_method = ""
        
        self.app = app

        self.signalSetup()
        self.loadUser()
        self.userSetup()
    
    def signalSetup(self):
        """
        Setup for signal connections
        """
        self.ui.b_save.clicked.connect(self.openSave)
        self.ui.b_vid.clicked.connect(self.openVideo)
        #self.ui.b_of.clicked.connect(self.openOf)
        #self.ui.b_depth.clicked.connect(self.openDepth)
        self.ui.b_run.clicked.connect(self.startRun)
        self.ui.b_colour.clicked.connect(self.pickColour)
        self.ui.b_ground_truth.clicked.connect(self.openGroundTruth)

        self.ui.t_fps.textChanged.connect(self.changeFps)
        self.ui.c_error_plot.stateChanged.connect(self.checkFiles)
        self.ui.c_speed_plot.stateChanged.connect(self.checkFiles)

        #self.ui.c_of.stateChanged.connect(self.checkVideoCreation)
        #self.ui.c_back_of.stateChanged.connect(self.checkVideoCreation)
        #self.ui.c_depth.stateChanged.connect(self.checkVideoCreation)
    

    def openGroundTruth(self):
        """Open file with ground truth values for speed
        """
        gt_dir = self.openFile(self.user["GT"], title="Load Ground Truth Data", 
                                file_filter="Numpy Files (*.npy);;All files (*)")
        if gt_dir != "":
            self.user["GT"] = gt_dir
            self.ui.l_ground_truth.setText("Load: " + self.splitPath(gt_dir)[-1])
        self.checkFiles()
        

    def changeFps(self):
        """
        Change fps for the created videos
        """
        check = re.search("[1-9][0-9]*", self.ui.t_fps.text())
        if check:
            num = check.group()
            fps = int(num)
            if fps > self.fps_limit:
                print("Warning: Too big number for fps. Falling back to {} fps.".format(self.fps_limit))
                fps = self.fps_limit
            self.fps = fps
            self.ui.t_fps.setText(str(fps))
        else:
            print("Wrong Input For Fps")
            self.ui.t_fps.setText("30")
            self.fps = 30


    def splitPath(self, path):
        """Split the given path based on filesystem
        
        Arguments:
            path {str  } -- path you want to split
        
        Returns:
            path splitted on / or \ -- only use \ on windows
        """
        sep = "/"
        if platform.system() == "Windows": 
            sep = "\\"

        return path.split(sep)

    def openSave(self):
        """Open directory to save results
        """
        save_dir = QFileDialog.getExistingDirectory(self, "Select a folder", self.user["Save"], QFileDialog.ShowDirsOnly)
        if save_dir != "":
            self.user["Save"] = save_dir
            self.ui.l_save.setText("Save to: " + save_dir)
            self.checkFiles()

    def checkFiles(self):
        """Checks if there are folders from previous runs, so we don't need to create them. 
        """
        self.of_exist = True if os.path.exists(os.path.join(self.user["Save"], "Of")) else False
        self.back_of_exist = True if os.path.exists(os.path.join(self.user["Save"], "Back_Of")) else False
        self.img_exist = True if os.path.exists(os.path.join(self.user["Save"], "Images")) else False
        self.depth_exist = True if os.path.exists(os.path.join(self.user["Save"], "Depth")) else False

        self.ui.c_error_plot.setEnabled(self.user["GT"] != "")
        self.ui.c_error_plot_video.setEnabled(self.ui.c_error_plot.isChecked())
        self.ui.c_speed_plot_video.setEnabled(self.ui.c_speed_plot.isChecked())

        if self.runRequirements():
            self.ui.b_run.setEnabled(True)
        else:
            self.ui.b_run.setEnabled(False)
    

    def runRequirements(self):
        """Basic requirements to start run
        
        Returns:
            bool -- returns true if the requirements are satisfied
        """
        ready = (self.user["Save"] != "" and self.user["Video"]) or self.img_exist
        return ready

    def openVideo(self):
        """Open video to analyze
        """
        fname = self.openFile(self.user["Video"])
        if fname != "":
            self.user["Video"] = fname
            name = self.splitPath(fname)[-1]
            self.ui.l_vid.setText("Load: " + name)
            self.vid_name = name.split(".")[0]
            self.checkFiles()

    def openOf(self):
        fname = self.openFile(self.user["Of"], file_filter="Python files (*.py);;All files (*)")
        self.user["Of"] = fname
        name = self.splitPath(fname)[-1]
        self.ui.l_of.setText("Load: " + name)
        self.checkFiles()

    def openDepth(self):
        fname = self.openFile(self.user["Depth"], file_filter="Python files (*.py);;All files (*)")
        self.user["Depth"] = fname
        name = self.splitPath(fname)[-1]
        self.ui.l_depth.setText("Load: " + name)
        self.checkFiles()

    def openFile(self, folder, title="Open Video", file_filter="Video Files (*.mp4 *.avi *.mkv);;All files (*)"):
        """Open QFileDialog with the given parameters, returns selected file
        
        Arguments:
            folder {str} -- path where the file dialog should start
        
        Keyword Arguments:
            title {str} -- title of the file dialog (default: {"Open Video"})
            file_filter {str} -- filter for file extensions (default: {"Video Files (*.mp4 *.avi *.mkv);;All files (*)"})
        
        Returns:
            [str] -- path to file
        """
        fname = QFileDialog.getOpenFileName(self, title, folder, file_filter)   
        
        return fname[0]
    
    def userSetup(self):
        """Label text setup if the user json is not empty
        """
        if self.user["Save"] == "":
            #self.ui.b_vid.setEnabled(False)
            #self.ui.b_of.setEnabled(False)
            #self.ui.b_depth.setEnabled(False)
            self.ui.b_run.setEnabled(False)
        else:
            self.ui.l_save.setText(self.user["Save"])
            #if self.user["Of"] != "":
            #    self.ui.l_of.setText(self.user["Of"])
            #if self.user["Depth"] != "":
            #    self.ui.l_depth.setText(self.user["Depth"])
            if self.user["Video"] != "":
                self.ui.l_vid.setText(self.user["Video"])
            if self.user["GT"] != "":
                self.ui.l_ground_truth.setText(self.splitPath(self.user["GT"])[-1])
            
            self.vid_name = self.splitPath(self.user["Video"])[-1]
            self.ui.l_colour.setText(self.user["Colour"])
            
            
    
    def loadUser(self):
        """Load user file if exists, create empty otherwise
        """ 
        if(os.path.isfile(self.user_file)):
            print("Found User File")
            with open(self.user_file, "r") as json_file:
                self.user = json.load(json_file)
            self.checkFiles()
        else:
            self.user = {"Save":"","Of":"","Depth":"","Video":"", "Colour":"#1a1a1b", "GT":""}
            self.saveUser()

    def saveUser(self):
        """Saves user data to json file
        """
        with open(self.user_file, "w+") as json_file:
            json.dump(self.user, json_file, indent=4)

    def errorChecks(self):
        stop_calculation = False
        #ret = QMessageBox.question(self, "Solving Errors", "Click a button", QMessageBox.Ok | QMessageBox.Abort, QMessageBox.Ok)
        found_error = False
        errors = {
            "Info": [],
            "Critical": []
        }
        ori_images = 0
        of_images = 0
        depth_images = 0
        back_of_images = 0

        if os.path.exists(self.savePathJoin("Images")):
            ori_images = len(list_directory(self.savePathJoin("Images"), extension="png"))
        #elif os.path.exists(self.user["Video"]):
        #    ori_images = count_frames(self.user["Video"])


        #print(count_frames(self.user["Video"]))
        print(os.path.exists(self.savePathJoin("Images")))

        # Check image folder
        if self.img_exist and not os.path.exists(self.savePathJoin("Images")):
            if os.path.exists(self.user["Video"]):
                errors["Info"].append("Images folder {0} doesn't exist -> Recreate it and recalculate optical flow and depth estimations".format(self.savePathJoin("Images")))
            else:
                stop_calculation = True
                errors["Critical"].append(("Images folder {0} and video file {1} don't exist -> Stopping run".format(self.savePathJoin("Images"), self.user["Video"])))
        elif self.img_exist and os.path.exists(self.user["Video"]):
            errors["Info"].append("Both video {0} and Images folder {1} exist -> using Images folder by default".format(self.user["Video"], self.savePathJoin("Images")))
            #ori_images_vid = count_frames(self.user["Video"])
            #
            #if ori_images_vid != ori_images:
            #    errors["Info"].append("Images in {0} with number {1} doesn't match the video {2} frame number {3} -> Recreating images and recalculating optical flow and depth estimations".
            #        format(self.savePathJoin("Images"), ori_images, self.user["Video"], ori_images_vid))
        elif not self.img_exist and os.path.exists(self.user["Video"]):
            errors["Info"].append("Images folder {0} doesn't exist -> Create it and calculate optical flow and depth estimations".format(self.savePathJoin("Images")))


        # Check video file
        if self.user["Video"] != "" and not os.path.exists(self.user["Video"]):
            if os.path.exists(self.savePathJoin("Images")):
                errors["Info"].append(("Video file {0} doesn't exist -> Using images in the Images folder instead".format(self.user["Video"])))
            else:
                stop_calculation = True
                errors["Critical"].append(("Images folder {0} and video file {1} don't exist -> Stopping run".format(self.savePathJoin("Images"), self.user["Video"])))
        elif os.path.exists(self.user["Video"]) and os.path.exists(self.savePathJoin("Images")):
            pass

        # Check optical flow
        if self.of_exist and not os.path.exists(self.savePathJoin("Of")):
            errors["Info"].append(("Optical flow folder {0} doesn't exist -> Recalculating optical flow".format(self.savePathJoin("Of"))))
        elif self.of_exist:
            of_images = len(list_directory(self.savePathJoin("Of"), extension="png"))
            if of_images != ori_images - 1 and ori_images != 0:
                errors["Info"].append(("Optical flow number {0} doesn't match video image number {1} - 1 -> Recalculating optical flow".format(of_images, ori_images)))



        #print(len(list_directory(self.savePathJoin("Images"), extension="png")))
        #print(len(list_directory(self.savePathJoin("Of"), extension="png")))
        #print(len(list_directory(self.savePathJoin("Depth"), extension="png")))        
        print(ori_images)



        for key in errors:
            print(key)
            for e in errors[key]:
                print("     " + str(e))

        return True




        if(not os.path.isfile(self.user["GT"]) and self.ui.l_ground_truth != ""):
            self.user["GT"] = ""
            errors.append("Ground Truth file doesn't exist -> file won't be used")

        

        return True


    def buildParamsDict(self):
        self.params_dict = {
            "img_dir": self.savePathJoin("Images"),
            "depth_dir": self.savePathJoin("Depth"),
            "back_of_dir": self.savePathJoin("Back_Of"),
            "of_dir": self.savePathJoin("Of"),
            "save_dir": self.user["Save"],
            "high": 1,
            "low": 0.309,
            "run_dict": self.run_dict,
            "of_model": self.app.get_resource(os.path.join("of_models", "network-default.pytorch")),
            "depth_model": self.app.get_resource(os.path.join("depth_models", "model_city2kitti.meta")),
            "plot_speed_dir": PLOT_SPEED_DIR,
            "numbers_dir": NP_DIR,
            "plot_error_dir": PLOT_ERROR_DIR,
            "speed_gt": self.user["GT"],
            "vid_path": self.user["Video"],
            "super_pixel_method": self.super_pixel_method
        }

    def startRun(self):
        """Start calculations
        """
        if self.errorChecks():
            return
        
        self.disableButtons()
        self.saveUser()
        self.sendUser.emit(self.user)
        self.buildCreatedDict()
        self.createDirs()
        print("Start Run")
        self.buildRunDict()
        self.buildParamsDict()
        self.showProgressBar()
        self.startCalcThread()
        
    
    def startCalcThread(self):
        """Starting calculations on another thread
        """
        # 1 - create Worker and Thread inside the Form
        self.worker = calcRunner.CalculationRunner(self.params_dict, self.savePathJoin("Images"),
            self.savePathJoin("Depth"), self.savePathJoin("Of"), self.savePathJoin("Back_Of"),
            self.user["Save"], None, 1, 0.309, self.run_dict, self.app.get_resource(os.path.join("of_models", "network-default.pytorch")),
            self.app.get_resource(os.path.join("depth_models", "model_city2kitti.meta")), PLOT_SPEED_DIR,
            NP_DIR, PLOT_ERROR_DIR, speed_gt=self.user["GT"], vid_path=self.user["Video"], super_pixel_method=self.super_pixel_method)  # no parent!
        self.thread = QThread()  # no parent!

        self.worker.labelUpdate.connect(self.labelUpdate)

        self.worker.update.connect(self.progressUpdate)
        #self.worker.updateFin.connect(self.progressAllUpdate)

        self.worker.finished.connect(self.finishThread)

        self.worker.moveToThread(self.thread)
        #self.worker.finished.connect(self.stopVideo)
        self.thread.started.connect(self.worker.startThread)
        # 6 - Start the thread
        self.thread.start()

    def progressUpdate(self, value):
        """Updating both progressbar
        
        Arguments:
            value {int} -- current progress
        """
        self.progressBar.setValue(value)
        self.run_count += 1
        print("Update to:",self.run_count)
        self.progressAllBar.setValue(self.run_count)

    def finishThread(self):
        """Clean up after calculations thread finished
        """
        print("Fin Thread")
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        self.accept()

    def labelUpdate(self, run_dict):
        """Update progressbar label's text to show the current calculation
        
        Arguments:
            run_dict {dictionary} -- dictionary of the current process
        """
        self.progressBar.reset()
        self.progressBar.setMinimum(1)
        self.progressBar.setMaximum(run_dict["Progress"])
        self.progressLabel.setText(run_dict["Text"])


    def showProgressBar(self):
        """Create two progressbars to show how the calculations progresses
        """
        print("Show progress bar")
        self.progressLabel = QLabel(self)
        font = QFont()
        font.setFamily("GE Inspira")
        font.setPointSize(20)
        self.progressLabel.setFont(font)
        self.progressLabel.setAlignment(Qt.AlignCenter)
        self.progressLabel.setText("Hello")
        self.ui.layout_v.addWidget(self.progressLabel)

        self.progressBar = QProgressBar(self) # Progress bar created
        self.ui.layout_v.addWidget(self.progressBar)

        all_run = sum([self.run_dict[key]["Progress"] for key in self.run_dict if self.run_dict[key]["Run"]])
        print("All run:", all_run)
        self.progressAllBar = QProgressBar(self) # Progress bar created
        self.progressAllBar.setMinimum(1)
        self.progressAllBar.setMaximum(all_run)
        self.ui.layout_v.addWidget(self.progressAllBar)
        self.progressAllBar.setValue(1)

    def buildCreatedDict(self):
        """Build dictionary containing the created plots (if there's any).
        Send signal with this dictionary
        """
        self.created = {}
        if self.ui.c_speed_plot.isChecked():
            self.created["Speed_Plot"] = self.savePathJoin(PLOT_SPEED_DIR)
        if self.ui.c_error_plot.isChecked():
            self.created["Error_Plot"] = self.savePathJoin(PLOT_ERROR_DIR)
        self.sendCreated.emit(self.created)


    def buildRunDict(self):
        """Build dictionary with all calculations
        Dictionary fields:
            -Run: {bool} true if needs to be calculated, false otherwise
            -Progress: {int} the amount of steps to complete the calculation (used to update the progressbar)
            -Text: {str} the text to be showed in the progressbar label's when the calculation is running   
        """
        ori_images = 0
        if self.img_exist:
            ori_images = len(list_directory(self.savePathJoin("Images")))
        else:
            pass
            #cam = cv2.VideoCapture(self.user["Video"])       
            #ret,frame = cam.read() 
            #while(ret):
            #    print ('Reading...{0}'.format(ori_images)) 
            #    ori_images += 1
            #    ret,frame = cam.read() 
            #cam.release() 
            #cv2.destroyAllWindows()

        self.run_dict["Video"] = {"Run": not self.img_exist, "Progress":ori_images, "Text":"Preparing video"}

        self.run_dict["Of"] = {"Run": not self.of_exist, "Progress":ori_images, "Text":"Running optical flow"}
        self.run_dict["Back_Of"] = {"Run": not self.back_of_exist, "Progress":ori_images, "Text":"Running back optical flow"}
        self.run_dict["Depth"] = {"Run": not self.depth_exist, "Progress":ori_images, "Text":"Running depth estimation"}
        self.run_dict["Speed"] = {"Run": True, "Progress":ori_images, "Text":"Running speed estimation"}

        self.run_dict["Of_Vid"] = {"Run": self.ui.c_of.isChecked(), "Progress":ori_images, "Text":"Creating optical flow video"}
        self.run_dict["Back_Of_Vid"] = {"Run": self.ui.c_back_of.isChecked(), "Progress":ori_images, "Text":"Creating backward optical flow video"}
        self.run_dict["Depth_Vid"] = {"Run": self.ui.c_depth.isChecked(), "Progress":ori_images, "Text":"Creating depth estimation video"}

        self.run_dict["Speed_Plot"] = {"Run": self.ui.c_speed_plot.isChecked(), "Progress":ori_images, "Text":"Creating plot for speed values"}
        self.run_dict["Error_Plot"] = {"Run": self.ui.c_error_plot.isChecked(), "Progress":ori_images, "Text":"Creating plot for speed error"}

        self.run_dict["Speed_Plot_Video"] = {"Run": self.ui.c_speed_plot_video.isChecked(), "Progress":ori_images, "Text":"Creating speed plot video"}
        self.run_dict["Error_Plot_Video"] = {"Run": self.ui.c_error_plot_video.isChecked(), "Progress":ori_images, "Text":"Creating error plot video"}


    def disableButtons(self):
        """Disable widgets while the calculation is running
        """
        self.setEnabled(False)
        #self.progressLabel.setEnabled(True)
        #self.ui.b_run.setEnabled(False)
        #self.ui.b_colour.setEnabled(False)
        #self.ui.b_ground_truth.setEnabled(False)
        ##self.ui.b_depth.setEnabled(False)
        ##self.ui.b_of.setEnabled(False)
        #self.ui.b_vid.setEnabled(False)
        #self.ui.b_save.setEnabled(False)

    def savePathJoin(self, path):
        """Join the given path with the save path (stored in user info)
        
        Arguments:
            path {str} -- path to join after the save path
        
        Returns:
            str -- joined path
        """
        return os.path.join(self.user["Save"], path)

    def createDir(self, dir_name):
        """Create directory with the given name in the save path (stored in user info)
        
        Arguments:
            dir_name {[type]} -- [description]
        """
        os.mkdir(os.path.join(self.user["Save"], dir_name))

    def createDirs(self):
        """Create or recreate (destroy and create) directories for the calculations
        """
        print("Creating Directories")

        #if not self.img_exist:
        #    self.createDir("Images")
        #if not self.of_exist:
        #    self.createDir("Of")
        #if not self.back_of_exist:
        #    self.createDir("Back_Of")
        #if not self.depth_exist:
        #    self.createDir("Depth")

        #self.reCreateDir(RESULTS)        
        #self.reCreateDir(OTHER_DIR)
        #self.reCreateDir(DRAW_DIR)
        #self.reCreateDir(SUPER_PIXEL_DIR)
        #self.reCreateDir(VL_DIR)
        #self.reCreateDir(NP_DIR)
        #self.reCreateDir(MASK_DIR)
        #if self.ui.c_speed_plot.isChecked():
        #    self.reCreateDir(PLOT_SPEED_DIR)
        #if self.user["GT"] != "" and self.ui.c_error_plot.isChecked():
        #    self.reCreateDir(PLOT_ERROR_DIR)

    def reCreateDir(self, name):
        """Create directory with the given name in save dir (stored in user info).
        If it already exists, then delete it first
        
        Arguments:
            name {str} -- name of the new directory
        """
        path = self.savePathJoin(name)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def pickColour(self):
        """Opens a QColorDialog to choose a colour
        """
        colour = QColorDialog.getColor()
        if colour.isValid():
            self.user["Colour"] = colour.name()
            self.ui.l_colour.setText(self.user["Colour"])