from PyQt5.QtWidgets import QDialog, QFileDialog, QColorDialog, QProgressBar, QLabel
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

import speed.pwc.run as pwc


RESULTS = 'results'
OTHER_DIR = os.path.join(RESULTS, 'other')
VL_DIR = os.path.join(RESULTS, 'velocity')
NP_DIR = os.path.join(RESULTS, 'numbers')
MASK_DIR = os.path.join(RESULTS, 'mask')


class Dialog(QDialog, Ui_Dialog):
    sendUser = pyqtSignal(object)

    def __init__(self, app, parent=None):
        super(Dialog, self).__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle("Process Video")
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.user_file = os.path.join(self.dir, ".userInfo.json") 
        self.user = None
        self.thread = None
        self.progressBar = None
        self.progressAllBar = None
        self.progressLabel = None
        self.img_exist = False
        self.of_exist = False
        self.back_of_exist = False
        self.depth_exist = False
        self.vid_name = None
        self.all_run = 1
        self.run_count = 1
        self.run_dict = {}
        
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
        self.ui.b_of.clicked.connect(self.openOf)
        self.ui.b_depth.clicked.connect(self.openDepth)
        self.ui.b_run.clicked.connect(self.startRun)
        self.ui.b_colour.clicked.connect(self.pickColour)
    
    def splitPath(self, path):
        sep = "/"
        if platform.system() == "Windows": 
            sep = "\\"

        return path.split(sep)

    def openSave(self):
        save_dir = QFileDialog.getExistingDirectory(self, "Select a folder", self.user["Save"], QFileDialog.ShowDirsOnly)
        self.user["Save"] = save_dir
        self.ui.l_save.setText("Save to: " + save_dir)
        self.checkFiles()

    def checkFiles(self):
        self.of_exist = True if os.path.exists(os.path.join(self.user["Save"], "Of")) else False
        self.back_of_exist = True if os.path.exists(os.path.join(self.user["Save"], "Back_Of")) else False
        self.img_exist = True if os.path.exists(os.path.join(self.user["Save"], "Images")) else False
        self.depth_exist = True if os.path.exists(os.path.join(self.user["Save"], "Depth")) else False

        if self.runRequirements():
            self.ui.b_run.setEnabled(True)
        else:
            self.ui.b_run.setEnabled(False)
    
    def runRequirements(self):
        ready = ((self.depth_exist         and self.of_exist         and self.img_exist)           or\
                (self.user["Depth"] != "" and self.of_exist         and self.img_exist)           or\
                (self.depth_exist         and self.user["Of"] != "" and self.img_exist)           or\
                (self.depth_exist         and self.of_exist         and self.user["Video"] != "") or\
                (self.user["Depth"] != "" and self.user["Of"] != "" and self.img_exist)           or\
                (self.depth_exist         and self.user["Of"] != "" and self.user["Video"] != "") or\
                (self.user["Depth"] != "" and self.of_exist         and self.user["Video"] != "") or\
                (self.user["Depth"] != "" and self.user["Of"] != "" and self.user["Video"] != "")) and\
                ((self.back_of_exist and self.of_exist) or self.user["Of"] != "")
        return ready

    def openVideo(self):
        fname = self.openFile(self.user["Video"])
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

    def openFile(self, folder, file_filter="Video Files (*.mp4 *.avi *.mkv);;All files (*)"):
        fname = QFileDialog.getOpenFileName(self, "Open Video", folder, file_filter)   
        
        return fname[0]
    
    def userSetup(self):
        if self.user["Save"] == "":
            #self.ui.b_vid.setEnabled(False)
            #self.ui.b_of.setEnabled(False)
            #self.ui.b_depth.setEnabled(False)
            self.ui.b_run.setEnabled(False)
        else:
            self.ui.l_save.setText(self.user["Save"])
            if self.user["Of"] != "":
                self.ui.l_of.setText(self.user["Of"])
            if self.user["Depth"] != "":
                self.ui.l_depth.setText(self.user["Depth"])
            if self.user["Video"] != "":
                self.ui.l_vid.setText(self.user["Video"])
    
    def loadUser(self):    
        if(os.path.isfile(self.user_file)):
            print("Found User File")
            with open(self.user_file, "r") as json_file:
                self.user = json.load(json_file)
            self.checkFiles()
            self.vid_name = self.splitPath(self.user["Video"])[-1]
            self.ui.l_colour.setText(self.user["Colour"])
        else:
            self.user = {"Save":"","Of":"","Depth":"","Video":"", "Colour":"#1a1a1b"}
            self.saveUser()

    def saveUser(self):
        with open(self.user_file, "w+") as json_file:
            json.dump(self.user, json_file, indent=4)

    def startRun(self):
        self.disableButtons()
        self.saveUser()
        self.sendUser.emit(self.user)
        self.createDirs()
        print("Start Run")
        self.createRunDict()
        self.showProgressBar()
        self.startCalcThread()
        
    
    def startCalcThread(self):
        # 1 - create Worker and Thread inside the Form
        self.worker = calcRunner.CalculationRunner(self.savePathJoin("Images"),
            self.savePathJoin("Depth"), self.savePathJoin("Of"), self.savePathJoin("Back_Of"),
            self.user["Save"], None, 1, 0.309, self.run_dict, self.app.get_resource("network-default.pytorch"))  # no parent!
        self.thread = QThread()  # no parent!

        self.worker.labelUpdate.connect(self.labelUpdate)

        self.worker.update.connect(self.progressUpdate)
        self.worker.updateFin.connect(self.progressAllUpdate)

        self.worker.finished.connect(self.finishThread)

        self.worker.moveToThread(self.thread)
        #self.worker.finished.connect(self.stopVideo)
        self.thread.started.connect(self.worker.startThread)
        # 6 - Start the thread
        self.thread.start()

    def progressUpdate(self, value):
        self.progressBar.setValue(value)
        self.run_count += 1
        self.progressAllBar.setValue(self.run_count)

    def finishThread(self):
        print("Fin Thread")
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        self.accept()

    def progressAllUpdate(self):
        pass
        #self.run_count += 1
        #self.progressAllBar.setValue(self.run_count)

    def labelUpdate(self, run_dict):
        self.progressBar.reset()
        self.progressBar.setMinimum(1)
        self.progressBar.setMaximum(run_dict["Progress"])
        self.progressLabel.setText(run_dict["Text"])


    def showProgressBar(self):
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

        self.progressAllBar = QProgressBar(self) # Progress bar created
        self.progressAllBar.setMinimum(1)
        self.progressAllBar.setMaximum(all_run)
        self.ui.layout_v.addWidget(self.progressAllBar)
        self.progressAllBar.setValue(1)

    def createRunDict(self):
        self.run_dict["Of"] = {"Run": not self.of_exist, "Progress":len(list_directory(self.savePathJoin("Images"))), "Text":"Running optical flow"}
        self.run_dict["Depth"] = {"Run": not self.depth_exist, "Progress":len(list_directory(self.savePathJoin("Images"))), "Text":"Running depth estimation"}
        self.run_dict["Speed"] = {"Run": True, "Progress":len(list_directory(self.savePathJoin("Images"))), "Text":"Running speed estimation"}

    def disableButtons(self):
        self.ui.b_run.setEnabled(False)
        self.ui.b_colour.setEnabled(False)
        self.ui.b_depth.setEnabled(False)
        self.ui.b_of.setEnabled(False)
        self.ui.b_vid.setEnabled(False)
        self.ui.b_save.setEnabled(False)

    def savePathJoin(self, path):
        return os.path.join(self.user["Save"], path)

    def createDir(self, dir_name):
        os.mkdir(os.path.join(self.user["Save"], dir_name))

    def createDirs(self):
        print("Creating Directories")

        if not self.img_exist:
            self.createDir("Images")
        if not self.of_exist:
            self.createDir("Of")
        if not self.back_of_exist:
            self.createDir("Back_Of")
        if not self.depth_exist:
            self.createDir("Depth")

        self.reCreateDir(RESULTS)        
        self.reCreateDir(OTHER_DIR)
        self.reCreateDir(VL_DIR)
        self.reCreateDir(NP_DIR)
        self.reCreateDir(MASK_DIR)

    def reCreateDir(self, name):
        path = self.savePathJoin(name)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def pickColour(self):
        colour = QColorDialog.getColor()
        if colour.isValid():
            self.user["Colour"] = colour.name()
            self.ui.l_colour.setText(self.user["Colour"])