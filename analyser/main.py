from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QObject

# https://github.com/pypa/setuptools/issues/1963
from mywindow import Ui_MainWindow
from PIL import Image
from PIL.ImageQt import ImageQt, toqpixmap
import videoPlayer
import utils
import speed.utils as sutils
import worker
import vid
import ImageHolder
import CycleVid
import skvideo.io
import sys
import re
import os
from dialog import Dialog
#from speed import speed_vectors


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, app, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Analyser")
        self.MAX_LEN = 10
        self.worker = None
        self.user = None
        self.vid_player = videoPlayer.VideoPlayer()
        self.ui.layout_vid.insertWidget(1, self.vid_player) # Insert video player into layout
        self.thread = None
        self.vid_opened = False
        self.vid_running = False
        self.fps_limit = 30
        self.images = None
        self.img_dir = None
        self.of_dir = None
        self.back_of_dir = None
        self.depth_dir = None
        self.app = app
        self.cycle_vid = CycleVid.CycleVid()      
        self.image_holder = ImageHolder.ImageHolder(self.MAX_LEN, self.fps_limit)
        self.ui.t_fps.setText(str(self.fps_limit))
        #self.vid_player.setScaledContents(True)        
        self.signalSetup()  
        self.disableSetup()

    def disableSetup(self):
        self.ui.b_video_left.setEnabled(False)
        self.ui.b_video_right.setEnabled(False)
        self.ui.b_video_up.setEnabled(False)
        self.ui.b_video_down.setEnabled(False)
        self.ui.actionPlay.setEnabled(False)
        self.ui.b_jump.setEnabled(False)
        self.ui.b_plot_left.setEnabled(False)
        self.ui.b_plot_right.setEnabled(False)
        self.ui.t_frame.setEnabled(False)
        self.ui.t_fps.setEnabled(False)

    def enableSetup(self):
        self.ui.b_video_left.setEnabled(True)
        self.ui.b_video_right.setEnabled(True)
        self.ui.b_video_up.setEnabled(True)
        self.ui.b_video_down.setEnabled(True)
        self.ui.actionPlay.setEnabled(True)
        self.ui.b_jump.setEnabled(True)
        self.ui.b_plot_left.setEnabled(True)
        self.ui.b_plot_right.setEnabled(True)
        self.ui.t_frame.setEnabled(True)
        self.ui.t_fps.setEnabled(True)

    def showDialog(self):
        widget = Dialog(app=self.app, parent=self)
        widget.sendUser.connect(self.changeUser)
        widget.rejected.connect(self.dialogExit)
        widget.accepted.connect(self.dialogAccept)
        widget.exec_()

    def dialogAccept(self):
        self.startSetup()
        self.enableSetup()

    def dialogExit(self):
        print("Exit")
        self.close()

    def startSetup(self):
        self.img_dir = os.path.join(self.user["Save"], "Images") 
        self.of_dir = os.path.join(self.user["Save"], "Of") 
        self.back_of_dir = os.path.join(self.user["Save"], "Back_Of") 
        self.depth_dir = os.path.join(self.user["Save"], "Depth")
        
        results = sutils.getResultDirs()
        velocity = os.path.join(self.user["Save"], results["Velocity"])
        mask = os.path.join(self.user["Save"], results["Mask"])

        self.cycle_vid.add("original", self.img_dir)
        self.cycle_vid.add("of", self.of_dir)
        self.cycle_vid.add("back_of", self.back_of_dir)
        self.cycle_vid.add("depth", self.depth_dir)
        self.cycle_vid.add("velocity", velocity)
        self.cycle_vid.add("mask", mask)
        self.openVideo(self.img_dir)

    def changeUser(self, user):
        self.user = user
        print("Changed Base Dir To:", self.user)

    def signalSetup(self):
        """
        Setup for signal connections
        """
        self.ui.b_video_right.clicked.connect(self.changeVideoToNextFrame)
        self.ui.b_video_left.clicked.connect(self.changeVideoToPrevFrame)
        self.ui.actionPlay.triggered.connect(self.startVideo)
        self.ui.t_fps.textChanged.connect(self.changeFps)
        self.ui.b_video_up.clicked.connect(self.cycleUp)
        self.ui.b_video_down.clicked.connect(self.cycleDown)
        self.ui.t_frame.textChanged.connect(self.changeFrameText)
        self.ui.b_jump.clicked.connect(self.jumpToFrame)
        self.ui.b_rerun.clicked.connect(self.showDialog)
        self.vid_player.resizeSignal.connect(self.resizeVideo)

    def changeFrameText(self):
        check = re.search("[1-9][0-9]*", self.ui.t_frame.text())
        if check:
            num = check.group()
            frame = int(num)
            maxF = self.image_holder.vidLen - 1
            if frame > maxF:
                print("Warning: Too big number for frame. Falling back to max {0} frame.".format(maxF))
                frame = maxF
            self.ui.t_frame.setText(str(frame))
        else:
            print("Wrong Input For Fps")
            self.ui.t_frame.setText("")

    def changeFps(self):
        """
        Change fps of video
        """
        check = re.search("[1-9][0-9]*", self.ui.t_fps.text())
        if check:
            num = check.group()
            fps = int(num)
            if fps > self.fps_limit:
                print("Warning: Too big number for fps. Falling back to {} fps.".format(self.fps_limit))
                fps = self.fps_limit
            self.image_holder.changeFps(fps)
            self.ui.t_fps.setText(str(fps))
        else:
            print("Wrong Input For Fps")
            self.ui.t_fps.setText("")

    def changeFrameTo(self, img):
        self.vid_player.setPixmap(toqpixmap(img))

    def resizeVideo(self, width, height):
        if self.vid_opened:
            img = self.image_holder.resize(width, height)
            self.changeFrameTo(img)

    def cycleUp(self):
        self.openVideo(self.cycle_vid.up(), self.image_holder.cur_idx)
        #img = self.image_holder.getCurrent()
        #img = self.image_holder.prepareImg(img)
        #self.image_holder.increment()
        #self.changeFrameTo(img)

    def cycleDown(self):
        self.openVideo(self.cycle_vid.down(), self.image_holder.cur_idx)
        #img = self.image_holder.getCurrent()
        #img = self.image_holder.prepareImg(img)
        #self.image_holder.increment()
        #self.changeFrameTo(img)

    def stopVideo(self):
        """
        Stop the video if it is running.
        Clean worker thread.
        """
        if self.vid_running:
            self.ui.actionPlay.setEnabled(False)
            print("Stop Video, Disable Play Button")
            self.vid_running = False
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()
            print("Enable Play Button")  
            print("Enable nextFrame Button")
            print("Enable prevFrame Button")          
            self.ui.actionPlay.setEnabled(True)
            self.ui.b_video_left.setEnabled(True)
            self.ui.b_video_right.setEnabled(True)
            self.ui.b_jump.setEnabled(True)
            self.ui.b_rerun.setEnabled(True)
            self.ui.b_plot_left.setEnabled(True)
            self.ui.b_plot_right.setEnabled(True)
            self.ui.t_fps.setEnabled(True)

    def startVideo(self): 
        """
        Starts video playing, or stops it if it is running.
        Starts video from the beginning if it is at it's end.
        Creates new worker thread for video playing.
        """    
        if self.vid_running:
            self.stopVideo()
        else:
            if self.image_holder.cur_idx >= self.image_holder.vidLen - 1:
                print("Reopen video from start")
                self.openVideo(self.cycle_vid.current())
            print("Start Video")
            print("Disable nextFrame Button")
            print("Disable prevFrame Button")
            self.vid_running = True
            self.ui.b_video_left.setEnabled(False)
            self.ui.b_video_right.setEnabled(False)
            self.ui.b_jump.setEnabled(False)
            self.ui.b_rerun.setEnabled(False)
            self.ui.b_plot_left.setEnabled(False)
            self.ui.b_plot_right.setEnabled(False)
            self.ui.t_fps.setEnabled(False)

            # 1 - create Worker and Thread inside the Form
            self.worker = worker.Worker(*self.image_holder.getStartData())  # no parent!
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

    def openVideo(self, img_dir, n_frame=None):
        self.vid_opened = True
        self.images = utils.readImg(img_dir)
        width = self.vid_player.width()
        height = self.vid_player.height()
        img = self.image_holder.setup(self.images, width, height, colour=self.user["Colour"], n_frame=n_frame)
        self.changeFrameTo(img)

    def jumpToFrame(self):
        n_frame = int(self.ui.t_frame.text())
        print("Jumping to frame: {0}".format(n_frame)) 
        self.image_holder.cur_idx = n_frame
        img = self.image_holder.jump(n_frame)
        self.changeFrameTo(img)

    def prevFrame(self):
        assert self.image_holder.cur_idx >= 0, ("Calling prevFrame at the beginning of video")

        #self.image_holder.cur_idx -= 1
        img = self.image_holder.prevImg()        

        return img

    def nextFrame(self):
        """
        Returns the next frame in the video as a PIL Image, resized into the size of the image holder
        keeping aspect ratios and filling the remaining place with some colour
        
        Returns:
            PIL Image -- The next frame of the video
        """
        assert self.vid_opened, ("Calling nextFrame before opening the video")
        assert self.image_holder.cur_idx < self.image_holder.vidLen - 1, ("Calling nextFrame at the end of video")

        #self.image_holder.cur_idx += 1
        img = self.image_holder.nextImg()        

        return img

    def changeVideoToNextFrame(self):
        """
        Display the next frame of the video
        """
        print("Checking: ", self.image_holder.cur_idx)
        if self.image_holder.cur_idx < self.image_holder.vidLen - 1:
            print("Success")
            self.vid_player.setPixmap(toqpixmap(self.nextFrame()))

    def changeVideoToPrevFrame(self):
        """
        Change video to previous frame.
        If there are images left in prev_frame then use them, otherwise jump to the previous frame.
        """
        cur = self.image_holder.cur_idx
        print("Checking: ", self.image_holder.cur_idx)
        if cur - 1 >= 0:
            print("Success")
            img = self.prevFrame()
            self.changeFrameTo(img)


if __name__ == '__main__':
    appctxt = ApplicationContext()
    #stylesheet = appctxt.get_resource('styles.qss')
    #appctxt.app.setStyleSheet(open(stylesheet).read())
    window = MainWindow(app=appctxt)
    window.show()
    exit_code = appctxt.app.exec_()
    #window.closeVid()
    sys.exit(exit_code)
    