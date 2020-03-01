from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QObject

# https://github.com/pypa/setuptools/issues/1963
from mywindow import Ui_MainWindow
from PIL import Image
from PIL.ImageQt import ImageQt, toqpixmap
from utils import *
import worker
import vid
import skvideo.io
import sys
import time



class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, app, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.ui.l_video.setPixmap(toqpixmap(self.nextFrame()))
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.cap = None
        self.worker = None
        self.thread = None
        self.vid_opened = False
        self.vid_running = False
        self.vid_data = vid.vidData()

        #self.player = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)
        #self.player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(app.get_resource("a.mp4"))))
        #self.player.setVideoOutput(self.ui.widget)
        #self.player.play()
        

        self.signalSetup()
        im = Image.open(app.get_resource("picasso.jpg"))
        pixImg = toqpixmap(im)
        self.ui.l_video.setPixmap(pixImg)
        self.mv = app.get_resource("b.mp4")
        self.openVideo(self.mv)
        img = self.nextFrame()
        p = toqpixmap(img)   


    def signalSetup(self):
        """
        Setup for signal connections
        """
        self.ui.b_video_right.clicked.connect(self.changeVideoToNextFrame)
        self.ui.b_video_left.clicked.connect(self.changeVideoToPrevFrame)
        self.ui.actionPlay.triggered.connect(self.startVideo)

    def startVideo(self):        
        if self.vid_running:
            self.stopVideo()
        else:
            print("Start Video")
            self.vid_running = True
            # 1 - create Worker and Thread inside the Form
            self.worker = worker.Worker(*self.vid_data.getStartData())  # no parent!
            self.thread = QThread()  # no parent!

            # 2 - Connect Worker`s Signals to Form method slots to post data.
            self.worker.intReady.connect(self.changeVideoToNextFrame)

            # 3 - Move the Worker object to the Thread object
            self.worker.moveToThread(self.thread)

            # 4 - Connect Worker Signals to the Thread slots
            self.worker.finished.connect(self.stopVideo)

            # 5 - Connect Thread started signal to Worker operational slot method
            self.thread.started.connect(self.worker.startCounting)

            # 6 - Start the thread
            self.thread.start()
    
    def stopVideo(self):
        if self.vid_running:
            self.ui.actionPlay.setEnabled(False)
            print("Stop Video, Disable Play Button")
            self.vid_running = False
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()
            print("Enable Play Button")            
            self.ui.actionPlay.setEnabled(True)
    
    def changeVideoToNextFrame(self):
        self.ui.l_video.setPixmap(toqpixmap(self.nextFrame()))
        self.vid_data.current_idx += 1

    def changeVideoToPrevFrame(self):
        cur = self.vid_data.current_idx
        if self.cur - 1 >= 0:
            self.cur -= 1
            if self.vid_opened:
                self.closeVid()
            self.openVideo(self.mv)
            for i in range(0,cur - 1):
                next(self.cap.nextFrame())
            self.ui.l_video.setPixmap(toqpixmap(Image.fromarray(next(self.cap.nextFrame()))))
    
    def openVideo(self, vid_path):
        """Opens the video and stores it's iterator in self.cap
        
        Arguments:
            vid_path {string} -- path to video
        """
        if self.vid_opened:
            print("Warning: Trying to open video that has already been opened")
            self.closeVid()
        self.cap = skvideo.io.FFmpegReader(vid_path)
        self.vid_opened = True
        self.vid_data.setup(self.cap.getShape())
    
    def nextFrame(self):
        """
        Returns the next frame in the video as a PIL Image
        
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
            self.vid_opened = False
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