from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QObject

# https://github.com/pypa/setuptools/issues/1963
from mywindow import Ui_MainWindow
from PIL import Image
from PIL.ImageQt import ImageQt, toqpixmap
from utils import *
import skvideo.io
import sys
import time

class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(int)

    @pyqtSlot()
    def procCounter(self): # A slot takes no params
        for i in range(0, 100):
            time.sleep(0.02)
            print(i)
            self.intReady.emit(i)
        self.finished.emit()

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, app, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.ui.l_video.setPixmap(toqpixmap(self.nextFrame()))
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.cap = None
        self.vid_opened = False

        #self.player = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)
        #self.player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(app.get_resource("a.mp4"))))
        #self.player.setVideoOutput(self.ui.widget)
        #self.player.play()
        
        self.signalSetup()
        im = Image.open(app.get_resource("picasso.jpg"))
        pixImg = toqpixmap(im)
        self.ui.l_video.setPixmap(pixImg)
        self.openVideo(app.get_resource("b.mp4"))
        img = self.nextFrame()
        p = toqpixmap(img)

        # 1 - create Worker and Thread inside the Form
        self.obj = Worker()  # no parent!
        self.thread = QThread()  # no parent!

        # 2 - Connect Worker`s Signals to Form method slots to post data.
        self.obj.intReady.connect(self.changeVideoToNextFrame)

        # 3 - Move the Worker object to the Thread object
        self.obj.moveToThread(self.thread)

        # 4 - Connect Worker Signals to the Thread slots
        self.obj.finished.connect(self.thread.quit)

        # 5 - Connect Thread started signal to Worker operational slot method
        self.thread.started.connect(self.obj.procCounter)

        # 6 - Start the thread
        self.thread.start()


    def signalSetup(self):
        """
        Setup for signal connections
        """
        self.ui.b_video_right.clicked.connect(self.changeVideoToNextFrame)
        self.ui.b_video_left.clicked.connect(self.changeVideoToNextFrame)

    
    def changeVideoToNextFrame(self):
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
    
    def closeVid(self):
        """
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
    window.closeVid()
    sys.exit(exit_code)