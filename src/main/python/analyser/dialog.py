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
        self.base_dir = None
        self.user = None
        
        self.app = app

        self.signalSetup()
        self.loadUser()
        self.userSetup()
    
    def signalSetup(self):
        """
        Setup for signal connections
        """
        self.ui.b_vid.clicked.connect(self.openVideo)

    def openVideo(self):
        file = QFileDialog.getOpenFileName(self, "Open Video", self.user["Video"], "Video Files (*.mp4 *.avi *.mkv);;All files (*)")
        
        sep = "/"
        if platform.system() == "Windows": 
            sep = "\\"

        fname = file[0].split(sep)[-1]
        self.ui.l_vid.setText("Load: " + fname)
        self.base_dir = fname.split(".")[0]
        self.baseDirChange.emit(self.base_dir)
    
    def userSetup(self):
        pass
    
    def loadUser(self):    
        if(os.path.isfile(self.user_file)):
            print("Open User File")
            with open(self.user_file, "r") as json_file:
                self.user = json.load(json_file)
        else:
            self.user = {"Save":"","Of":"","Depth":"","Video":""}
            with open(self.user_file, "w+") as json_file:
                json.dump(self.user, json_file, indent=4)