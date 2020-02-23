import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from mywindow import Ui_MainWindow
from PIL import Image
from PIL.ImageQt import ImageQt, toqpixmap
from utils import *
import skvideo.io
    

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.cap = None
        self.vid_opened = False
        # uic.loadUi("mainwindow.ui", self)
        im = Image.open("picasso.jpg")
        pixImg = toqpixmap(im)
        self.ui.l_video.setPixmap(pixImg)
        self.openVideo("a.mp4")
        self.nextFrame().show()
    
    def openVideo(self, vid_path):
        """Opens the video and stores it's iterator in self.cap
        
        Arguments:
            vid_path {string} -- path to video
        """
        self.cap = skvideo.io.FFmpegReader(vid_path)
        self.vid_opened = True
    
    def nextFrame(self):
        """Returns the next frame in the video as a PIL Image
        
        Returns:
            PIL Image -- The next frame of the video
        """
        assert self.vid_opened, ("Calling nextFrame before opening the video")
        return Image.fromarray(next(self.cap.nextFrame()))
    
    def closeVideo(self):
        """
        Closes the video if it was opened
        """
        if self.vid_opened:
            self.cap.close()
        


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()

window.show()
window.closeVideo()
sys.exit(app.exec_())