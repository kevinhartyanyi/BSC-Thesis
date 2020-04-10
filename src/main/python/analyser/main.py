from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QMainWindow, QAction, QApplication, QStyle
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, QSize

# https://github.com/pypa/setuptools/issues/1963
from mainwindowUI import Ui_MainWindow
from PIL import Image
from PIL.ImageQt import toqpixmap
import videoPlayer
import utils
import speed.utils as sutils
from PyQt5.QtCore import pyqtSlot
import worker
import qdarkgraystyle
import imageHolder
import cycleVid
import skvideo.io
import sys
import re
import os
from dialog import Dialog
from mainwindowInfo import MainWindowInfo
from logger import LogInfo
import logging


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
        self.vid_player.setToolTip("Video player")
        self.ui.layout_vid.insertWidget(1, self.vid_player) # Insert video player into layout
        self.plot_player = videoPlayer.VideoPlayer()
        self.plot_player.setToolTip("Plot player")        
        self.ui.layout_plot.insertWidget(1, self.plot_player)
        self.thread = None
        self.vid_opened = False
        self.vid_running = False
        self.fps_limit = 30
        self.images = None
        self.img_dir = None
        self.of_dir = None
        self.back_of_dir = None
        self.depth_dir = None
        self.created = None
        self.app = app
        self.cycle_vid = cycleVid.cycleVid() 
        self.cycle_plot = cycleVid.cycleVid()  
        self.image_holder = imageHolder.imageHolder(self.MAX_LEN, self.fps_limit)
        self.plot_holder = imageHolder.imageHolder(self.MAX_LEN, self.fps_limit)
        self.ui.t_fps.setText(str(self.fps_limit))    
        self.logInfo = LogInfo(parent=self)
        logging.info("Start App")
        self.imageSetup()
        self.signalSetup()  
        self.disableSetup()
    
    def imageSetup(self):
        """Load the button images and display them as an icon
        """
        left_arrow = QtGui.QPixmap(self.app.get_resource("left_arrow.png"))
        right_arrow = QtGui.QPixmap(self.app.get_resource("right_arrow.png"))
        up_arrow = QtGui.QPixmap(self.app.get_resource("up_arrow.png"))
        down_arrow = QtGui.QPixmap(self.app.get_resource("down_arrow.png"))
        self.ui.b_video_left.setIcon(QtGui.QIcon(self.app.get_resource("left_arrow.png")))
        self.ui.b_video_right.setIcon(QtGui.QIcon(self.app.get_resource("right_arrow.png")))
        self.ui.b_video_up.setIcon(QtGui.QIcon(self.app.get_resource("up_arrow.png")))
        self.ui.b_video_down.setIcon(QtGui.QIcon(self.app.get_resource("down_arrow.png")))
        self.ui.b_plot_left.setIcon(QtGui.QIcon(self.app.get_resource("left_arrow.png")))
        self.ui.b_plot_right.setIcon(QtGui.QIcon(self.app.get_resource("right_arrow.png")))

        self.ui.b_info.setIconSize(QSize(50,50))
        self.ui.b_info.setIcon(QApplication.style().standardIcon(QStyle.SP_MessageBoxInformation))

    def changeDescription(self):
        """Change the video description based on the current video
        """
        vid_type = self.cycle_vid.currentType()
        
        if vid_type == "original":
            self.ui.l_description.setText("The original video")
        elif vid_type == "of":
            self.ui.l_description.setText("Optical flow (motion of image objects between two consecutive frames)")
        elif vid_type == "back_of":
            self.ui.l_description.setText("Backward optical flow (optical flow with reversed frames)")
        elif vid_type == "depth":
            self.ui.l_description.setText("Depth estimation (darker means farther, brighter means closer)")
        elif vid_type == "velocity":
            self.ui.l_description.setText("Optical flow directions with colours (after throwing away the inconsistent optical flow)")
        elif vid_type == "mask":
            self.ui.l_description.setText("Speed mask (the coloured pixels are used in the speed estimation)")
        elif vid_type == "draw":
            self.ui.l_description.setText("Optical flow directions with arrows (after throwing away the inconsistent optical flow)")
        elif vid_type == "super_pixel":
            self.ui.l_description.setText("Speed values in the super pixel segmentations")
        elif vid_type == "object_detection":
            self.ui.l_description.setText("Object detection with YOLO model")

    def disableSetup(self):
        """Disable widgets on startup (before calculations)
        """
        self.ui.b_video_left.setEnabled(False)
        self.ui.b_video_right.setEnabled(False)
        self.ui.b_video_up.setEnabled(False)
        self.ui.b_video_down.setEnabled(False)
        self.ui.actionPlay.setEnabled(False)
        self.ui.actionDepth.setEnabled(False)
        self.ui.actionOF.setEnabled(False)
        self.ui.actionOFArrows.setEnabled(False)
        self.ui.actionOFDirections.setEnabled(False)
        self.ui.actionMask.setEnabled(False)
        self.ui.actionOriginal.setEnabled(False)
        self.ui.actionSuperPixel.setEnabled(False)
        self.ui.actionBackOF.setEnabled(False)
        self.ui.actionObjectDetection.setEnabled(False)
        self.ui.b_jump.setEnabled(False)
        self.ui.b_plot_left.setEnabled(False)
        self.ui.b_plot_right.setEnabled(False)
        self.ui.t_frame.setEnabled(False)
        self.ui.t_fps.setEnabled(False)

    def enableSetup(self):
        """Enable widgets after the calculations finished
        """
        self.ui.b_video_left.setEnabled(True)
        self.ui.b_video_right.setEnabled(True)
        self.ui.b_video_up.setEnabled(True)
        self.ui.b_video_down.setEnabled(True)
        self.ui.actionPlay.setEnabled(True)
        self.ui.actionOF.setEnabled(True)
        self.ui.actionDepth.setEnabled(True)
        self.ui.actionMask.setEnabled(True)
        self.ui.actionOriginal.setEnabled(True)
        self.ui.actionBackOF.setEnabled(True)
        self.ui.b_jump.setEnabled(True)
        self.ui.t_frame.setEnabled(True)
        self.ui.t_fps.setEnabled(True)

    def showInfo(self):
        """Show information dialog
        """
        widget = MainWindowInfo(app=self.app, parent=self)
        widget.exec_()

    def showDialog(self):
        """Open dialog where the user can choose the run options
        """
        widget = Dialog(app=self.app, parent=self)
        widget.sendUser.connect(self.changeUser)
        widget.sendCreated.connect(self.setCreated)
        widget.rejected.connect(self.dialogExit)
        widget.accepted.connect(self.dialogAccept)
        widget.exec_()

    def dialogAccept(self):
        """Start setup if the dialog was accepted
        """
        self.startSetup()
        self.enableSetup()

    def dialogExit(self):
        """Exit from the application
        """
        logging.info("Exit Run Dialog")
        self.close()

    def setCreated(self, created):
        """Check if there were any plots created, and load them
        
        Arguments:
            created {dictionary} -- Dictionary with the created plots
        """
        self.cycle_plot.reset()
        self.ui.b_plot_left.setEnabled(False)
        self.ui.b_plot_right.setEnabled(False)
        self.created = created

        if len(created) == 0:
            self.created = None
            return
        
        self.ui.b_plot_left.setEnabled(True)
        self.ui.b_plot_right.setEnabled(True)
        
        for key in created:
            if created[key] != "":
                self.cycle_plot.add(key, created[key])
                logging.info("Found Plot: {0}".format(created[key]))

    def startSetup(self):
        """Setup paths for data, load images into video players
        """
        self.img_dir = os.path.join(self.user["Save"], "Images") 
        self.of_dir = os.path.join(self.user["Save"], "Of") 
        self.back_of_dir = os.path.join(self.user["Save"], "Back_Of") 
        self.depth_dir = os.path.join(self.user["Save"], "Depth")
        self.obj_det_dir = os.path.join(self.user["Save"], "ObjectDetection")
        
        results = sutils.getResultDirs()
        velocity = os.path.join(self.user["Save"], results["Velocity"])
        mask = os.path.join(self.user["Save"], results["Mask"])
        draw = os.path.join(self.user["Save"], results["Draw"])
        super_pixel = os.path.join(self.user["Save"], results["SuperPixel"])

        self.vid_player.clear()
        self.plot_player.clear()

        self.cycle_vid.reset()
        self.cycle_vid.add("original", self.img_dir)
        self.cycle_vid.add("of", self.of_dir)
        self.cycle_vid.add("back_of", self.back_of_dir)
        self.cycle_vid.add("depth", self.depth_dir)
        self.cycle_vid.add("mask", mask)
        if os.path.exists(velocity):
            self.cycle_vid.add("velocity", velocity)
            self.ui.actionOFDirections.setEnabled(True)
        if os.path.exists(draw):
            self.cycle_vid.add("draw", draw)
            self.ui.actionOFArrows.setEnabled(True)
        if os.path.exists(super_pixel):
            self.cycle_vid.add("super_pixel", super_pixel)       
            self.ui.actionSuperPixel.setEnabled(True)
        if os.path.exists(self.obj_det_dir):
            self.cycle_vid.add("object_detection", self.obj_det_dir)
            self.ui.actionObjectDetection.setEnabled(True)

        plot_dir = None
        if self.created != None:
            plot_dir = self.cycle_plot.current()
        self.changeDescription()
        self.openVideo(self.img_dir, plot_dir=plot_dir)

    def changeUser(self, user):
        """Load user information from the dialog
        
        Arguments:
            user {dictionary} -- User information dictionary
        """
        self.user = user

    def signalSetup(self):
        """
        Setup for signal connections
        """
        self.ui.b_video_right.clicked.connect(self.changeVideoToNextFrame)
        self.ui.b_video_left.clicked.connect(self.changeVideoToPrevFrame)
        self.ui.b_info.clicked.connect(self.showInfo)
        self.ui.actionPlay.triggered.connect(self.startVideo)
        self.ui.actionOF.triggered.connect(self.cycleToSelected)
        self.ui.actionDepth.triggered.connect(self.cycleToSelected)
        self.ui.actionOriginal.triggered.connect(self.cycleToSelected)
        self.ui.actionOFDirections.triggered.connect(self.cycleToSelected)
        self.ui.actionOFArrows.triggered.connect(self.cycleToSelected)
        self.ui.actionSuperPixel.triggered.connect(self.cycleToSelected)
        self.ui.actionMask.triggered.connect(self.cycleToSelected)
        self.ui.actionBackOF.triggered.connect(self.cycleToSelected)
        self.ui.actionObjectDetection.triggered.connect(self.cycleToSelected)
        self.ui.actionShow_Log.triggered.connect(self.showLog)
        self.ui.actionInformation.triggered.connect(self.showInfo)
        self.ui.t_fps.textChanged.connect(self.changeFps)
        self.ui.b_video_up.clicked.connect(self.cycleUp)
        self.ui.b_video_down.clicked.connect(self.cycleDown)
        self.ui.t_frame.textChanged.connect(self.changeFrameText)
        self.ui.b_jump.clicked.connect(self.jumpToFrame)
        self.ui.b_rerun.clicked.connect(self.showDialog)
        self.ui.b_plot_left.clicked.connect(self.cyclePlotLeft)
        self.ui.b_plot_right.clicked.connect(self.cyclePlotRight)
        self.vid_player.resizeSignal.connect(self.resizeVideo)
        self.plot_player.resizeSignal.connect(self.resizePlotVideo)

    def showLog(self):
        """Show log information
        """
        widget = self.logInfo
        widget.exec_()

    def cycleToSelected(self):
        """Cycle to the selected video based on the action which triggered this function
        """
        action = self.sender()
        text = action.text()
        logging.info("Cycle To: {0}".format(text))
        if text == "OF":
            self.openVideo(self.cycle_vid.get("of"), n_frame=self.image_holder.cur_idx)
        elif text == "Depth":
            self.openVideo(self.cycle_vid.get("depth"), n_frame=self.image_holder.cur_idx)
        elif text == "Original":
            self.openVideo(self.cycle_vid.get("original"), n_frame=self.image_holder.cur_idx)
        elif text == "BackOF":
            self.openVideo(self.cycle_vid.get("back_of"), n_frame=self.image_holder.cur_idx)
        elif text == "OFDirections":
            self.openVideo(self.cycle_vid.get("velocity"), n_frame=self.image_holder.cur_idx)
        elif text == "OFArrows":
            self.openVideo(self.cycle_vid.get("draw"), n_frame=self.image_holder.cur_idx)
        elif text == "Mask":
            self.openVideo(self.cycle_vid.get("mask"), n_frame=self.image_holder.cur_idx)
        elif text == "SuperPixel":
            self.openVideo(self.cycle_vid.get("super_pixel"), n_frame=self.image_holder.cur_idx)
        elif text == "ObjectDetection":
            self.openVideo(self.cycle_vid.get("object_detection"), n_frame=self.image_holder.cur_idx)
        self.changeDescription()

    def cyclePlotLeft(self):
        """Change plot (cycle left in the plot_holder)
        """
        self.openVideo(plot_dir=self.cycle_plot.up(), n_frame=self.plot_holder.cur_idx)

    def cyclePlotRight(self):
        """Change plot (cycle right in the plot_holder)
        """
        self.openVideo(plot_dir=self.cycle_plot.down(), n_frame=self.plot_holder.cur_idx)

    def changeFrameText(self):
        """Check if the text typed into frame line edit is correct, if so then store the value as int
        """
        check = re.search("[1-9][0-9]*", self.ui.t_frame.text())
        if check:
            num = check.group()
            frame = int(num)
            maxF = self.image_holder.vidLen - 1
            if frame > maxF:
                logging.warning("Too big number for frame. Falling back to max {0} frame.".format(maxF))
                frame = maxF
            self.ui.t_frame.setText(str(frame))
        else:
            logging.info("Wrong Input For Frame")
            self.ui.t_frame.setText("0")

    def changeFps(self):
        """
        Change fps of video
        """
        check = re.search("[1-9][0-9]*", self.ui.t_fps.text())
        if check:
            num = check.group()
            fps = int(num)
            if fps > self.fps_limit:
                logging.warning("Too big number for fps. Falling back to {} fps.".format(self.fps_limit))
                fps = self.fps_limit
            self.image_holder.changeFps(fps)
            self.ui.t_fps.setText(str(fps))
        else:
            logging.info("Wrong Input For Fps")
            self.ui.t_fps.setText("30")

    def changeFrameTo(self, img, plot_img=None):
        """Change the currently displayed image on the video player or on the plot player
        
        Arguments:
            img {PIL Image} -- Display this image on the video player
        
        Keyword Arguments:
            plot_img {PIL Image} -- Display this image on the plot player (default: {None})
        """
        if img != None:
            self.vid_player.setPixmap(toqpixmap(img))
        if plot_img != None:
            self.plot_player.setPixmap(toqpixmap(plot_img))

    def resizeVideo(self, width, height):
        """Resize the image inside video player
        
        Arguments:
            width {int} -- new image width
            height {int} -- new image height
        """
        if self.vid_opened:
            img = self.image_holder.resize(width, height)
            self.changeFrameTo(img)
    
    def resizePlotVideo(self, width, height):
        """Resize the image inside plot player
        
        Arguments:
            width {int} -- new image width
            height {int} -- new image height
        """
        if self.vid_opened and self.created is not None:
            plot_img = self.plot_holder.resize(width, height)
            self.changeFrameTo(None, plot_img=plot_img)

    def cycleUp(self):
        """Cycle up in the video holder and display the video there
        """
        self.openVideo(self.cycle_vid.up(), self.image_holder.cur_idx)
        self.changeDescription()

    def cycleDown(self):
        """Cycle down in the video holder and display the video there
        """
        self.openVideo(self.cycle_vid.down(), self.image_holder.cur_idx)
        self.changeDescription()

    def stopVideo(self):
        """
        Stop the video if it is running.
        Clean worker thread.
        """
        if self.vid_running:
            self.ui.actionPlay.setEnabled(False)
            logging.info("Stop Video, Disable Play Button")
            self.vid_running = False
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()
            logging.info("Enable Play Button")         
            self.ui.actionPlay.setEnabled(True)
            self.ui.b_video_left.setEnabled(True)
            self.ui.b_video_right.setEnabled(True)
            self.ui.b_jump.setEnabled(True)
            self.ui.b_rerun.setEnabled(True)
            self.ui.b_plot_left.setEnabled(True)
            self.ui.b_plot_right.setEnabled(True)
            self.ui.t_fps.setEnabled(True)
            self.ui.t_frame.setEnabled(True)

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
                logging.info("Reopen video from start")
                plot_dir = None
                if self.created != None:
                    plot_dir = self.cycle_plot.current()
                self.openVideo(self.cycle_vid.current(), plot_dir=plot_dir)
            logging.info("Start Video")
            self.vid_running = True
            self.ui.b_video_left.setEnabled(False)
            self.ui.b_video_right.setEnabled(False)
            self.ui.b_jump.setEnabled(False)
            self.ui.b_rerun.setEnabled(False)
            self.ui.b_plot_left.setEnabled(False)
            self.ui.b_plot_right.setEnabled(False)
            self.ui.t_fps.setEnabled(False)
            self.ui.t_frame.setEnabled(False)

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

    def openVideo(self, img_dir=None, n_frame=None, plot_dir=None):
        """Open video/plot and jump to the frame number (if given)
        
        Keyword Arguments:
            img_dir {str} -- Path where the images are located (for the video player) (default: {None})
            n_frame {int} -- Frame number from where the video should start from (default: {None})
            plot_dir {str} -- Path where the images are located (for the plot player) (default: {None})
        """
        if n_frame is not None and n_frame < 0:
            n_frame = 0
        img = None
        if img_dir is not None:
            self.vid_opened = True
            self.images = sutils.list_directory(img_dir, extension=".png")
            width = self.vid_player.width()
            height = self.vid_player.height()
            img = self.image_holder.setup(self.images, width, height, colour=self.user["Colour"], n_frame=n_frame)
        
        plot_img = None
        if plot_dir != None:
            plots = sutils.list_directory(plot_dir, extension=".png")
            logging.info("Plots: {0}".format(len(plots)))
            width = self.plot_player.width()
            height = self.plot_player.height()
            plot_img = self.plot_holder.setup(plots, width, height, colour=self.user["Colour"], n_frame=n_frame)
        self.changeFrameTo(img, plot_img)

    def jumpToFrame(self):
        """Jumps to the frame that was given in the frame line edit (t_frame) by the user
        """
        n_frame = int(self.ui.t_frame.text())
        logging.info("Jumping to frame: {0}".format(n_frame)) 
        self.image_holder.cur_idx = n_frame
        img = self.image_holder.jump(n_frame)
        plot_img = None
        if self.created != None:
            plot_img = self.plot_holder.jump(n_frame)
        self.changeFrameTo(img, plot_img)

    def prevFrame(self):
        """
        Returns the previous frame in the video/plot as a PIL Image, resized into the size of the image holder
        keeping aspect ratios and filling the remaining place with some colour
        
        Returns:
            (PIL Image, PIL Image) -- Tuple: video image and plot image (plot image is None if there are no plots)
        """
        assert self.image_holder.cur_idx >= 0, ("Calling prevFrame at the beginning of video")

        img = self.image_holder.prevImg()  
        plot_img = None
        if self.created != None:
            plot_img = self.plot_holder.prevImg()      

        return img, plot_img

    def nextFrame(self):
        """
        Returns the next frame in the video/plot as a PIL Image, resized into the size of the image holder
        keeping aspect ratios and filling the remaining place with some colour
        
        Returns:
            (PIL Image, PIL Image) -- Tuple: video image and plot image (plot image is None if there are no plots)
        """
        assert self.vid_opened, ("Calling nextFrame before opening the video")
        assert self.image_holder.cur_idx < self.image_holder.vidLen - 1, ("Calling nextFrame at the end of video")

        img = self.image_holder.nextImg()  
        plot_img = None
        if self.created != None:
            plot_img = self.plot_holder.nextImg()      

        return img, plot_img

    def changeVideoToNextFrame(self):
        """
        Display the next frame of the video/plot
        """
        logging.info("Checking: {0}".format(self.image_holder.cur_idx))
        if self.image_holder.cur_idx < self.image_holder.vidLen - 1:
            img, plot_img = self.nextFrame()
            self.vid_player.setPixmap(toqpixmap(img))
            if plot_img != None:
                self.plot_player.setPixmap(toqpixmap(plot_img))

    def changeVideoToPrevFrame(self):
        """
        Display the previous frame of the video/plot
        """
        cur = self.image_holder.cur_idx
        logging.info("Checking: {0}".format(self.image_holder.cur_idx))
        if cur - 1 >= 0:
            img, plot_img = self.prevFrame()
            self.changeFrameTo(img, plot_img)


if __name__ == '__main__':
    appctxt = ApplicationContext()
    appctxt.app.setStyleSheet(qdarkgraystyle.load_stylesheet())
    window = MainWindow(app=appctxt)
    window.show()
    exit_code = appctxt.app.exec_()
    sys.exit(exit_code)
    