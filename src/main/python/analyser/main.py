from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets

from mywindow import Ui_MainWindow
from PIL import Image
from PIL.ImageQt import ImageQt, toqpixmap
from utils import *
import skvideo.io
import cv2

import sys

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, app, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.cap = None
        self.vid_opened = False
        
        self.signalSetup()
        #im = Image.open("picasso.jpg")
        im = Image.open(app.get_resource("picasso.jpg"))
        pixImg = toqpixmap(im)
        self.ui.l_video.setPixmap(pixImg)
        self.openVideo(app.get_resource("a.mp4"))
        img = self.nextFrame()
        p = toqpixmap(img)
        #self.ui.l_video.setPixmap(toqpixmap(self.nextFrame()))


    def signalSetup(self):
        """
        Setup for signal connections
        """
        self.ui.b_video_right.clicked.connect(self.changeVideoToNextFrame)

    
    def changeVideoToNextFrame(self):
        print("Hello")
        self.ui.l_video.setPixmap(toqpixmap(self.nextFrame()))

    
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
    
    def __del__(self):
        """
        Destructor:
        Closes the video if it was opened
        """
        if self.vid_opened:
            self.cap.close()


if __name__ == '__main__':
    appctxt = ApplicationContext()
    #stylesheet = appctxt.get_resource('styles.qss')
    #appctxt.app.setStyleSheet(open(stylesheet).read())
    window = MainWindow(app=appctxt)
    window.show()
    exit_code = appctxt.app.exec_()
    sys.exit(exit_code)