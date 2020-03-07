from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtCore import pyqtSignal
from dialogUI import Ui_Dialog
import platform
import json
import os



class Dialog(QDialog, Ui_Dialog):
    baseDirChange = pyqtSignal(str)

    def __init__(self, app, parent=None):
        super(Dialog, self).__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle("Process Video")
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.user_file = os.path.join(self.dir, ".userInfo.json") 
        self.user = None
        self.img_exist = False
        self.of_exist = False
        self.depth_exist = False
        
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
    
    def splitPath(self, path):
        sep = "/"
        if platform.system() == "Windows": 
            sep = "\\"

        return path.split(sep)

    def openSave(self):
        save_dir = QFileDialog.getExistingDirectory(self, "Select a folder", self.user["Save"], QFileDialog.ShowDirsOnly)
        self.user["Save"] = save_dir
        self.ui.l_vid.setText("Save to: " + save_dir)
        self.checkFiles()

    def checkFiles(self):
        self.of_exist = True if os.path.exists(os.path.join(self.user["Save"], "Of")) else False
        self.img_exist = True if os.path.exists(os.path.join(self.user["Save"], "Pictures")) else False
        self.depth_exist = True if os.path.exists(os.path.join(self.user["Save"], "Depth")) else False

        if self.runRequirements():
            self.ui.b_run.setEnabled(True)
        else:
            self.ui.b_run.setEnabled(False)
    
    def runRequirements(self):
        ready = (self.depth_exist         and self.of_exist         and self.img_exist)           or\
                (self.user["Depth"] != "" and self.of_exist         and self.img_exist)           or\
                (self.depth_exist         and self.user["Of"] != "" and self.img_exist)           or\
                (self.depth_exist         and self.of_exist         and self.user["Video"] != "") or\
                (self.user["Depth"] != "" and self.user["Of"] != "" and self.img_exist)           or\
                (self.depth_exist         and self.user["Of"] != "" and self.user["Video"] != "") or\
                (self.user["Depth"] != "" and self.of_exist         and self.user["Video"] != "") or\
                (self.user["Depth"] != "" and self.user["Of"] != "" and self.user["Video"] != "")
        return ready
    def openVideo(self):
        fname = self.openFile(self.user["Video"])
        self.user["Video"] = fname
        name = self.splitPath(fname)[-1]
        self.ui.l_vid.setText("Load: " + name)
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
            self.ui.b_vid.setEnabled(False)
            self.ui.b_of.setEnabled(False)
            self.ui.b_depth.setEnabled(False)
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
        else:
            self.user = {"Save":"","Of":"","Depth":"","Video":""}
            self.saveUser()

    def saveUser(self):
        with open(self.user_file, "w+") as json_file:
            json.dump(self.user, json_file, indent=4)

    def startRun(self):
        self.saveUser()
        self.baseDirChange.emit(self.user["Save"])
        self.createDirs()
        self.accept()

    def createDir(self, dir_name):
        os.mkdir(os.path.join(self.user["Save"], dir_name))

    def createDirs(self):
        if not self.img_exist:
            self.createDir("Images")
        if not self.of_exist:
            self.createDir("Of")
        if not self.depth_exist:
            self.createDir("Depth")