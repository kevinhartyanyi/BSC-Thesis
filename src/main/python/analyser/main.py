from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QObject

# https://github.com/pypa/setuptools/issues/1963
from mywindow import Ui_MainWindow
from PIL import Image
from PIL.ImageQt import ImageQt, toqpixmap
import utils
import worker
import vid
import frameHolder
import imageHolder
import cycleVid
import skvideo.io
import sys
import time
import re
#from speed import speed_vectors


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, app, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.MAX_LEN = 10
        self.cap = None
        self.worker = None
        self.thread = None
        self.vid_opened = False
        self.vid_running = False
        self.images = None
        self.cycle_vid = cycleVid.cycleVid()
        self.vid_data = vid.vidData()
        self.prev_frames = frameHolder.frameHolder(self.MAX_LEN)        
        self.image_holder = imageHolder.imageHolder(self.MAX_LEN)
        #self.ui.l_video.setScaledContents(True)

        self.signalSetup()
        self.mv = app.get_resource("data/vid_0001/Pictures")
        self.mv2 = app.get_resource("data/vid_0001/2011_09_26_drive_0001.mp4")
        self.openVideo_new(self.mv)
        self.of = app.get_resource("data/vid_0001/pwc.mp4")
        self.cycle_vid.add("original", self.mv)
        self.cycle_vid.add("opticalFlow", self.of)

        #self.openVideo(self.mv)


    def signalSetup(self):
        """
        Setup for signal connections
        """
        self.ui.b_video_right.clicked.connect(self.changeVideoToNextFrame_new)
        self.ui.b_video_left.clicked.connect(self.changeVideoToPrevFrame_new)
        self.ui.actionPlay.triggered.connect(self.startVideo_new)
        self.ui.t_fps.textChanged.connect(self.changeFps)
        self.ui.b_video_up.clicked.connect(self.cycleUp)
        self.ui.b_video_down.clicked.connect(self.cycleDown)

    def changeFps(self):
        """
        Change fps of video
        """
        check = re.search("[1-9][0-9]*", self.ui.t_fps.text())
        if check:
            num = check.group()
            fps = int(num)
            if fps > 50:
                print("Warning: Too big number for fps. Falling back to 50 fps.")
                fps = 50
            self.vid_data.fps = fps
            self.ui.t_fps.setText(num)
        else:
            print("Wrong Input For Fps")
            self.ui.t_fps.setText("")

    def cycleUp(self):
        vid = self.cycle_vid.up()
        self.closeVid()
        self.openVideo(vid)

    def cycleDown(self):
        vid = self.cycle_vid.down()
        self.closeVid()
        self.openVideo(vid)

    def startVideo(self): 
        """
        Starts video playing, or stops it if it is running.
        Starts video from the beginning if it is at it's end.
        Creates new worker thread for video playing.
        """    
        if self.vid_running:
            self.stopVideo()
        else:
            if self.vid_data.current_idx >= self.vid_data.end_idx - 1:
                print("Reopen video from start")
                self.closeVid()
                self.openVideo(self.mv)
            print("Start Video")
            print("Disable nextFrame Button")
            print("Disable prevFrame Button")
            self.vid_running = True
            self.ui.b_video_left.setEnabled(False)
            self.ui.b_video_right.setEnabled(False)
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

    def changeVideoToNextFrame(self):
        """
        Display the next frame of the video
        """
        if self.vid_data.current_idx < self.vid_data.end_idx - 1:
            self.ui.l_video.setPixmap(toqpixmap(self.nextFrame()))

    def changeVideoToPrevFrame(self):
        """
        Change video to previous frame.
        If there are images left in prev_frame then use them, otherwise jump to the previous frame.
        """
        cur = self.vid_data.current_idx
        print("Cur Frame",cur)
        if cur - 1 > 0:
            cur -= 1
            self.vid_data.current_idx = cur
            img, success = self.prev_frames.prev()
            if success:
                self.ui.l_video.setPixmap(toqpixmap(img))
            else:
                print("No more elements are left in frameHolder. Reopening video.")
                self.jumpToFrame(cur, self.mv)

    def jumpToFrame(self, n_frame, video):  
        """
        Open video and jump to the specified frame
        
        Arguments:
            n_frame {int} -- frame number to jump to
            video {string} -- video path
        """
        print("Jumping to frame: {0}".format(n_frame))    
        if self.vid_opened:
            self.closeVid()
        self.openVideo(self.mv)
        for i in range(0, n_frame - 1):
            self.nextFrame()
        self.ui.l_video.setPixmap(toqpixmap(self.nextFrame()))
    
    def openVideo(self, vid_path):
        """
        Opens the video and stores it's iterator in self.cap 
        
        Arguments:
            vid_path {string} -- path to video
        """
        if self.vid_opened:
            print("Warning: Trying to open video that has already been opened")
            self.closeVid()
        self.cap = skvideo.io.FFmpegReader(vid_path)
        self.vid_opened = True
        self.vid_data.setup(self.cap.getShape())
        self.prev_frames.reset()
        self.changeVideoToNextFrame()

    def nextFrame(self):
        """
        Returns the next frame in the video as a PIL Image, resized into the size of the image holder
        keeping aspect ratios and filling the remaining place with some colour
        
        Returns:
            PIL Image -- The next frame of the video
        """
        assert self.vid_opened, ("Calling nextFrame before opening the video")
        assert self.vid_data.current_idx < self.vid_data.end_idx - 1, ("Calling nextFrame at the end of video")

        img = None
        image, success = self.prev_frames.nextI()
        if success:
            img = image
        else:
            width = self.ui.l_video.width()
            height = self.ui.l_video.height()
            img = utils.resizeImg(Image.fromarray(next(self.cap.nextFrame())), width)
            img = utils.fillImg(img, size=(width, height))
        
            self.prev_frames.add(img)

        self.vid_data.current_idx += 1

        return img
    
    def closeVid(self):
        """
        Closes the video if it was opened
        """
        if self.vid_opened:
            self.vid_opened = False
            self.cap.close()

    def startVideo_new(self): 
        """
        Starts video playing, or stops it if it is running.
        Starts video from the beginning if it is at it's end.
        Creates new worker thread for video playing.
        """    
        if self.vid_running:
            self.stopVideo()
        else:
            if self.vid_data.current_idx >= self.vid_data.end_idx - 1:
                print("Reopen video from start")
                self.openVideo_new(self.mv)
            print("Start Video")
            print("Disable nextFrame Button")
            print("Disable prevFrame Button")
            self.vid_running = True
            self.ui.b_video_left.setEnabled(False)
            self.ui.b_video_right.setEnabled(False)
            # 1 - create Worker and Thread inside the Form
            self.worker = worker.Worker(*self.vid_data.getStartData())  # no parent!
            self.thread = QThread()  # no parent!

            # 2 - Connect Worker`s Signals to Form method slots to post data.
            self.worker.intReady.connect(self.changeVideoToNextFrame_new)

            # 3 - Move the Worker object to the Thread object
            self.worker.moveToThread(self.thread)

            # 4 - Connect Worker Signals to the Thread slots
            self.worker.finished.connect(self.stopVideo)

            # 5 - Connect Thread started signal to Worker operational slot method
            self.thread.started.connect(self.worker.startCounting)

            # 6 - Start the thread
            self.thread.start()

    def openVideo_new(self, img_dir):
        if self.vid_opened:
            print("Warning: Trying to open video that has already been opened")
        self.vid_opened = True
        self.images = utils.readImg(img_dir)
        width = self.ui.l_video.width()
        height = self.ui.l_video.height()
        self.image_holder.setup(self.images, width, height)
        self.vid_data.setupWithList(self.images)

    def prevFrame_new(self):
        self.vid_data.current_idx -= 1
        img = Image.open(self.images[self.vid_data.current_idx - 1])
        width = self.ui.l_video.width()
        height = self.ui.l_video.height()
        img = utils.resizeImg(img, width)
        img = utils.fillImg(img, size=(width, height))

        return img

    def nextFrame_new(self):
        """
        Returns the next frame in the video as a PIL Image, resized into the size of the image holder
        keeping aspect ratios and filling the remaining place with some colour
        
        Returns:
            PIL Image -- The next frame of the video
        """
        assert self.vid_opened, ("Calling nextFrame before opening the video")
        assert self.vid_data.current_idx < self.vid_data.end_idx - 1, ("Calling nextFrame at the end of video")

        self.vid_data.current_idx += 1
        width = self.ui.l_video.width()
        height = self.ui.l_video.height()
        img = self.image_holder.nextImg(width, height)        

        return img

    def changeVideoToNextFrame_new(self):
        """
        Display the next frame of the video
        """
        if self.vid_data.current_idx < self.vid_data.end_idx - 1:
            self.ui.l_video.setPixmap(toqpixmap(self.nextFrame_new()))

    def changeVideoToPrevFrame_new(self):
        """
        Change video to previous frame.
        If there are images left in prev_frame then use them, otherwise jump to the previous frame.
        """
        cur = self.vid_data.current_idx
        print("Cur Frame",cur)
        if cur - 1 > 0:
            cur -= 1
            self.vid_data.current_idx = cur
            img = self.prevFrame_new()
            self.ui.l_video.setPixmap(toqpixmap(img))


if __name__ == '__main__':
    appctxt = ApplicationContext()
    #stylesheet = appctxt.get_resource('styles.qss')
    #appctxt.app.setStyleSheet(open(stylesheet).read())
    window = MainWindow(app=appctxt)
    window.show()
    exit_code = appctxt.app.exec_()
    window.closeVid()
    sys.exit(exit_code)