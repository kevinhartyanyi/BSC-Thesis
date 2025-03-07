from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QColorDialog,
    QProgressBar,
    QLabel,
    QMessageBox,
    QApplication,
    QStyle,
)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, QSize
from View.Run.dialogUI import Ui_Dialog
from View.Run.dialogInfo import DialogInfo
import platform
import json
import Model.Run.calcRunner as calcRunner
import os
import re
import shutil
from Model.Algorithms.utils import listDirectory, getResultDirs
import logging
import cv2

result_dir = getResultDirs()
RESULTS = result_dir["Results"]
OTHER_DIR = result_dir["Other"]
VL_DIR = result_dir["Velocity"]
NP_DIR = result_dir["Numbers"]
MASK_DIR = result_dir["Mask"]
DRAW_DIR = result_dir["Draw"]
SUPER_PIXEL_DIR = result_dir["SuperPixel"]
PLOT_SPEED_DIR = result_dir["Plot_Speed"]
PLOT_ERROR_DIR = result_dir["Plot_Error"]
PLOT_CRASH_DIR = result_dir["Plot_Crash"]


class Dialog(QDialog, Ui_Dialog):
    sendUser = pyqtSignal(object)
    sendCreated = pyqtSignal(object)

    def __init__(self, app, parent=None):
        super(Dialog, self).__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle("Process Video")
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.created = {}
        self.user = None
        self.thread = None
        self.progressBar = None
        self.progressAllBar = None
        self.progressLabel = None
        self.img_exist = False
        self.params_dict = None
        self.of_exist = False
        self.back_of_exist = False
        self.depth_exist = False
        self.gt_exist = False
        self.create_super_pixel_label = False
        self.no_error = True
        self.object_detection_dir_exist = False
        self.vid_name = None
        self.all_run = 1
        self.run_count = 1
        self.fps = 30
        self.fps_limit = 60
        self.low = 0.0
        self.high = 1.0
        self.run_dict = {}
        self.super_pixel_method = ""
        self.app = app
        self.ui.b_info.setIconSize(QSize(50, 50))
        self.ui.b_info.setIcon(
            QApplication.style().standardIcon(QStyle.SP_MessageBoxInformation)
        )
        homedir = os.path.expanduser("~")

        if platform.system() == "Windows":
            datadir = os.sep.join([homedir, "Analyser"])
        else:
            datadir = os.sep.join([homedir, ".analyser"])

        if not os.path.exists(datadir):
            os.makedirs(datadir)

        self.user_file = os.path.join(datadir, ".userInfo.json")

        self.signalSetup()
        self.loadUser()
        self.userSetup()

    def signalSetup(self):
        """
        Setup for signal connections
        """
        self.ui.b_info.clicked.connect(self.showInfo)
        self.ui.b_save.clicked.connect(self.openSave)
        self.ui.b_vid.clicked.connect(self.openVideo)
        self.ui.b_run.clicked.connect(self.startRun)
        self.ui.b_colour.clicked.connect(self.pickColour)
        self.ui.b_ground_truth.clicked.connect(self.openGroundTruth)

        self.ui.t_fps.textChanged.connect(self.changeFps)
        self.ui.t_low.editingFinished.connect(self.changeLow)
        self.ui.t_high.editingFinished.connect(self.changeHigh)
        self.ui.c_error_plot.stateChanged.connect(self.checkFiles)
        self.ui.c_speed_plot.stateChanged.connect(self.checkFiles)
        self.ui.c_crash_plot.stateChanged.connect(self.checkFiles)
        self.ui.combo_superpixel.currentIndexChanged.connect(
            self.changeSuperPixelMethod
        )
        self.ui.c_optimize.stateChanged.connect(self.checkFiles)
        self.ui.c_draw.stateChanged.connect(self.checkFiles)
        self.ui.c_velocity.stateChanged.connect(self.checkFiles)
        self.ui.c_object_detection.stateChanged.connect(self.checkFiles)

    def showInfo(self):
        """Show information dialog
        """
        widget = DialogInfo(app=self.app, parent=self)
        widget.exec_()

    def changeLow(self):
        """Change the value of the low drop based on the text inside t_low
        """
        self.changeLowHigh(self.ui.t_low, t_type="low")

    def changeHigh(self):
        """Change the value of the high drop based on the text inside t_high
        """
        self.changeLowHigh(self.ui.t_high, t_type="high")

    def changeLowHigh(self, text_widget, t_type="low"):
        """Change the value of low/high drop
        
        Arguments:
            text_widget {QLineEdit} -- t_low or t_high
        
        Keyword Arguments:
            t_type {str} -- which line edit was given (default: {"low"})
        """
        check = re.search("(0[.][0-9]+|1)", text_widget.text())
        if check and self.ui.t_low.text() != self.ui.t_high.text():
            num = check.group()
            i_num = float(num)
            if t_type == "low":
                self.low = i_num
            else:
                self.high = i_num
            text_widget.setText(str(i_num))
        else:
            logging.info("Wrong Input For low or high")
            if t_type == "low":
                text_widget.setText("0.0")
                self.low = 0.0
            else:
                text_widget.setText("1")
                self.high = 1

    def changeSuperPixelMethod(self, index):
        """Change the super pixel method
        
        Arguments:
            index {int} -- index of the selected super pixel method
        """
        if index == 0:
            self.super_pixel_method = ""
        elif index == 1:
            self.super_pixel_method = "Felzenszwalb"
        elif index == 2:
            self.super_pixel_method = "Quickshift"
        elif index == 3:
            self.super_pixel_method = "Slic"
        elif index == 4:
            self.super_pixel_method = "Watershed"

        self.checkFiles()

    def openGroundTruth(self):
        """Open file with ground truth values for speed
        """
        gt_dir = self.openFile(
            self.user["GT"],
            title="Load Ground Truth Data",
            file_filter="Numpy Files (*.npy)",
        )
        if gt_dir != "":
            self.user["GT"] = gt_dir
            self.ui.l_ground_truth.setText("Load: " + self.splitPath(gt_dir)[-1])
        self.checkFiles()

    def changeFps(self):
        """
        Change fps for the created videos
        """
        check = re.search("[1-9][0-9]*", self.ui.t_fps.text())
        if check:
            num = check.group()
            fps = int(num)
            if fps > self.fps_limit:
                logging.warning(
                    "Too big number for fps. Falling back to {0} fps.".format(
                        self.fps_limit
                    )
                )
                fps = self.fps_limit
            self.fps = fps
            self.ui.t_fps.setText(str(fps))
        else:
            logging.info("Wrong Input For Fps")
            self.ui.t_fps.setText("30")
            self.fps = 30

    def splitPath(self, path):
        """Split the given path based on filesystem
        
        Arguments:
            path {str} -- path you want to split
        
        Returns:
            path splitted on / or \ -- only uses \ on windows
        """
        return os.path.split(path)

    def openSave(self):
        """Open directory to save results
        """
        save_dir = QFileDialog.getExistingDirectory(
            self, "Select a folder", self.user["Save"], QFileDialog.ShowDirsOnly
        )
        if save_dir != "":
            self.user["Save"] = save_dir
            name_split = self.splitPath(save_dir)[-1]
            name = name_split.split(".")[0]
            self.ui.l_save.setText("Save to: " + name)
            self.checkFiles()

    def checkFiles(self):
        """Checks if there are folders from previous runs, so we don't need to create them.
        Also, checks condotions for checkboxes and buttons 
        """
        if self.user["Save"] != "":
            self.of_exist = os.path.exists(os.path.join(self.user["Save"], "Of"))
            self.back_of_exist = os.path.exists(
                os.path.join(self.user["Save"], "Back_Of")
            )
            self.img_exist = os.path.exists(os.path.join(self.user["Save"], "Images"))
            self.depth_exist = os.path.exists(os.path.join(self.user["Save"], "Depth"))

            self.object_detection_dir_exist = os.path.exists(
                os.path.join(self.user["Save"], "ObjectDetection")
            )

        self.gt_exist = self.user["GT"] != ""

        self.create_super_pixel_label = (
            self.super_pixel_method != ""
            and not os.path.exists(
                os.path.join(self.savePathJoin("Super_Pixel"), self.super_pixel_method)
            )
        )

        self.ui.c_crash_plot_video.setEnabled(self.ui.c_crash_plot.isChecked())
        self.ui.t_low.setEnabled(not self.ui.c_optimize.isChecked())
        self.ui.t_high.setEnabled(not self.ui.c_optimize.isChecked())
        self.ui.c_optimize.setEnabled(self.gt_exist)
        self.ui.c_error_plot.setEnabled(self.gt_exist)
        self.ui.c_error_plot_video.setEnabled(self.ui.c_error_plot.isChecked())
        self.ui.c_speed_plot_video.setEnabled(self.ui.c_speed_plot.isChecked())
        self.ui.c_super_pixel_video.setEnabled(
            self.ui.combo_superpixel.currentIndex() != 0
        )
        self.ui.c_csv.setEnabled(self.ui.c_error_plot.isChecked())

        if self.runRequirements():
            self.ui.b_run.setEnabled(True)
        else:
            self.ui.b_run.setEnabled(False)

    def runRequirements(self):
        """Basic requirements to start run
        
        Returns:
            bool -- returns true if the requirements are satisfied
        """
        ready = (self.user["Save"] != "" and self.user["Video"] != "") or self.img_exist
        return ready

    def openVideo(self):
        """Open video to analyze
        """
        fname = self.openFile(self.user["Video"])
        if fname != "":
            error_opening_video = False

            cam = cv2.VideoCapture(fname)
            logging.info("Opening video Check: {0}".format(fname))

            currentframe = 0
            ret, frame = cam.read()

            if ret is False:
                error_opening_video = True
            # Release all space and windows once done
            cam.release()
            cv2.destroyAllWindows()

            if error_opening_video:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("Error on opening the video: {0}".format(fname))
                msg.setWindowTitle("Information")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            else:
                self.user["Video"] = fname
                name = self.splitPath(fname)[-1]
                self.vid_name = name.split(".")[0]
                self.ui.l_vid.setText("Load: " + self.vid_name)
                self.checkFiles()

    def openFile(
        self, folder, title="Open Video", file_filter="Video Files (*.mp4 *.avi *.mkv)"
    ):
        """Open QFileDialog with the given parameters, returns selected file
        
        Arguments:
            folder {str} -- path where the file dialog should start
        
        Keyword Arguments:
            title {str} -- title of the file dialog (default: {"Open Video"})
            file_filter {str} -- filter for file extensions (default: {"Video Files (*.mp4 *.avi *.mkv)"})
        
        Returns:
            str -- path to file
        """
        fname = QFileDialog.getOpenFileName(self, title, folder, file_filter)

        return fname[0]

    def userSetup(self):
        """Label text setup if the user json is not empty
        """
        if self.user["Save"] == "":
            self.ui.b_run.setEnabled(False)
        else:
            name_split = self.splitPath(self.user["Save"])[-1]
            name = name_split.split(".")[0]
            self.ui.l_save.setText("Save to: " + name)

            if self.user["GT"] != "":
                self.ui.l_ground_truth.setText(self.splitPath(self.user["GT"])[-1])

            self.ui.l_colour.setText(self.user["Colour"])

    def loadUser(self):
        """Load user file if exists, create empty otherwise
        """
        if os.path.isfile(self.user_file):
            logging.info("Found User File")
            with open(self.user_file, "r") as json_file:
                self.user = json.load(json_file)
            self.checkFiles()
        else:
            self.user = {
                "Save": "",
                "Of": "",
                "Depth": "",
                "Video": "",
                "Colour": "#1a1a1b",
                "GT": "",
            }
            self.saveUser()

    def saveUser(self):
        """Saves user data to json file
        """
        self.user["Video"] = ""
        with open(self.user_file, "w+") as json_file:
            json.dump(self.user, json_file, indent=4)

    def errorChecks(self):
        """Check for errors and inform the user in a QMessageBox if found any
        
        Returns:
            bool -- if true then the execution stops
        """
        stop_calculation = False
        found_error = False
        errors = {"Info": [], "Critical": []}
        error_types = []
        ori_images = 0
        of_images = 0
        depth_images = 0
        back_of_images = 0

        if os.path.exists(self.savePathJoin("Images")):
            ori_images = len(
                listDirectory(self.savePathJoin("Images"), extension="png")
            )
        # Check image folder
        if self.img_exist and not os.path.exists(self.savePathJoin("Images")):
            if os.path.exists(self.user["Video"]):
                errors["Info"].append(
                    "Images folder {0} doesn't exist -> Recreate it and recalculate optical flow and depth estimations".format(
                        self.savePathJoin("Images")
                    )
                )
                error_types.append("NoImages")
            else:
                stop_calculation = True
                errors["Critical"].append(
                    (
                        "Images folder {0} and video file {1} don't exist -> Stopping run".format(
                            self.savePathJoin("Images"), self.user["Video"]
                        )
                    )
                )
        elif self.img_exist and os.path.exists(self.user["Video"]):
            errors["Info"].append(
                "Both the video {0} and Images folder {1} exist -> using Images folder by default".format(
                    self.user["Video"], self.savePathJoin("Images")
                )
            )
        elif not self.img_exist and not os.path.isfile(self.user["Video"]):
            stop_calculation = True
            errors["Critical"].append(
                (
                    "Images folder {0} and video file {1} don't exist -> Stopping run".format(
                        self.savePathJoin("Images"), self.user["Video"]
                    )
                )
            )

        # Check video file
        if self.user["Video"] != "" and not os.path.isfile(self.user["Video"]):
            if os.path.exists(self.savePathJoin("Images")):
                errors["Info"].append(
                    (
                        "Video file {0} doesn't exist -> Using images in the Images folder instead".format(
                            self.user["Video"]
                        )
                    )
                )
            else:
                stop_calculation = True
                errors["Critical"].append(
                    (
                        "Images folder {0} and video file {1} don't exist -> Stopping run".format(
                            self.savePathJoin("Images"), self.user["Video"]
                        )
                    )
                )
        elif os.path.isfile(self.user["Video"]) and os.path.exists(
            self.savePathJoin("Images")
        ):
            pass

        # Check optical flow
        if self.of_exist and not os.path.exists(self.savePathJoin("Of")):
            errors["Info"].append(
                (
                    "Optical flow folder {0} doesn't exist -> Recalculating optical flow".format(
                        self.savePathJoin("Of")
                    )
                )
            )
            error_types.append("NoOf")
        elif self.of_exist:
            of_images = len(listDirectory(self.savePathJoin("Of"), extension="png"))
            if of_images != ori_images - 1 and ori_images != 0:
                errors["Info"].append(
                    (
                        "Optical flow image number {0} doesn't match video image number {1} - 1 -> Recalculating optical flow".format(
                            of_images, ori_images
                        )
                    )
                )
                error_types.append("NoOf")

        # Check backward optical flow
        if self.back_of_exist and not os.path.exists(self.savePathJoin("Back_Of")):
            errors["Info"].append(
                (
                    "Backward optical flow folder {0} doesn't exist -> Recalculating backward optical flow".format(
                        self.savePathJoin("Back_Of")
                    )
                )
            )
            error_types.append("NoOf")
        elif self.back_of_exist:
            back_of_images = len(
                listDirectory(self.savePathJoin("Back_Of"), extension="png")
            )
            if back_of_images != of_images:
                errors["Info"].append(
                    (
                        "Backward optical flow image number {0} doesn't match optical flow image number {1} -> Recalculating backward optical flow".format(
                            back_of_images, of_images
                        )
                    )
                )
                error_types.append("NoOf")

        # Check depth estimation
        if self.depth_exist and not os.path.exists(self.savePathJoin("Depth")):
            errors["Info"].append(
                (
                    "Depth folder {0} doesn't exist -> Recalculating depth".format(
                        self.savePathJoin("Depth")
                    )
                )
            )
            error_types.append("NoDepth")
        elif self.depth_exist:
            depth_images = len(
                listDirectory(self.savePathJoin("Depth"), extension="png")
            )
            if depth_images != ori_images and ori_images != 0:
                errors["Info"].append(
                    (
                        "Depth image number {0} doesn't match video image number {1} -> Recalculating depth".format(
                            depth_images, ori_images
                        )
                    )
                )
                error_types.append("NoDepth")

        # Check ground truth
        if self.gt_exist and not os.path.isfile(self.user["GT"]):
            errors["Info"].append(
                (
                    "Ground Truth file {0} doesn't exist -> File won't be used".format(
                        self.user["GT"]
                    )
                )
            )
            error_types.append("NoGT")

        # Check super pixel labels
        if (
            self.super_pixel_method != ""
            and os.path.exists(
                os.path.join(self.savePathJoin("Super_Pixel"), self.super_pixel_method)
            )
            and ori_images != 0
            and len(
                listDirectory(
                    os.path.join(
                        self.savePathJoin("Super_Pixel"), self.super_pixel_method
                    ),
                    extension=".npy",
                )
            )
            != ori_images
        ):
            errors["Info"].append(
                (
                    "Super pixel label number {0} doesn't match image number {1} -> Recalculating super pixel labels".format(
                        len(
                            listDirectory(
                                os.path.join(
                                    self.savePathJoin("Super_Pixel"),
                                    self.super_pixel_method,
                                ),
                                extension=".npy",
                            )
                        ),
                        ori_images,
                    )
                )
            )
            error_types.append("LabelError")

        # Check object detection
        if self.ui.c_object_detection.isChecked() and os.path.exists(
            self.savePathJoin("ObjectDetection")
        ):
            if (
                len(
                    listDirectory(
                        self.savePathJoin("ObjectDetection"), extension=".png"
                    )
                )
                != ori_images
            ):
                errors["Info"].append(
                    "Object Detection image number {0} doesn't match image number of video {1} -> Recalculating object detection".format(
                        len(
                            listDirectory(
                                self.savePathJoin("ObjectDetection"), extension=".png"
                            )
                        ),
                        ori_images,
                    )
                )
                error_types.append("ObDetError")
            elif (
                len(
                    listDirectory(
                        self.savePathJoin("ObjectDetection"), extension=".npy"
                    )
                )
                != ori_images
            ):
                errors["Info"].append(
                    "Object Detection numpy array number {0} doesn't match image number of video {1} -> Recalculating object detection".format(
                        len(
                            listDirectory(
                                self.savePathJoin("ObjectDetection"), extension=".npy"
                            )
                        ),
                        ori_images,
                    )
                )
                error_types.append("ObDetError")

        answer = ""
        if len(errors["Info"]) > 0 and len(errors["Critical"]) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(
                "Some calculations might not run the way you expect them.\nIn show details check the right side of the arrows to see what will happen."
            )
            msg.setWindowTitle("Information")
            all_info = ""
            for info in errors["Info"]:
                all_info += info + "\n\n"
            msg.setDetailedText(all_info)
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Abort)
            answer = msg.exec_()
        elif len(errors["Critical"]) > 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText(
                "Found critical error\nCouldn't start run, see show details for more information"
            )
            msg.setWindowTitle("Critical Error")
            all_info = ""
            for info in errors["Critical"]:
                all_info += info + "\n"
            msg.setDetailedText(all_info)
            msg.setStandardButtons(QMessageBox.Abort)
            answer = msg.exec_()

        if answer != int("0x00040000", 16):
            for ty in error_types:
                logging.info("Solve error: {0}".format(ty))
                if ty == "NoImage":
                    self.img_exist = False
                    self.of_exist = False
                    self.back_of_exist = False
                    self.depth_exist = False
                elif ty == "NoOf":
                    self.of_exist = False
                    self.back_of_exist = False
                elif ty == "NoDepth":
                    self.depth_exist = False
                elif ty == "NoGT":
                    self.gt_exist = False
                    self.user["GT"] = ""
                elif ty == "LabelError":
                    self.create_super_pixel_label = True
                    shutil.rmtree(
                        os.path.join(
                            self.savePathJoin("Super_Pixel"), self.super_pixel_method
                        )
                    )
                elif ty == "ObDetError":
                    self.object_detection_dir_exist = False
                    shutil.rmtree(self.savePathJoin("ObjectDetection"))

        return answer == int("0x00040000", 16) or stop_calculation

    def calculateError(self, errorMessage):
        """Called if there were errors while calculating the results, creates QMessageBox
        
        Arguments:
            errorMessage {str} -- error message to be shown in the QMessageBox
        """
        self.no_error = False
        QMessageBox.warning(
            self, "Found Error", errorMessage, QMessageBox.Ok, QMessageBox.Ok
        )

    def buildParamsDict(self):
        """Build parameter dictionary to forward parameters for the calcRunner class
        """
        self.params_dict = {
            "img_dir": self.savePathJoin("Images"),
            "depth_dir": self.savePathJoin("Depth"),
            "back_of_dir": self.savePathJoin("Back_Of"),
            "of_dir": self.savePathJoin("Of"),
            "save_dir": self.user["Save"],
            "high": self.high,
            "low": self.low,
            "run_dict": self.run_dict,
            "of_model": self.app.get_resource(
                os.path.join("of_models", "network-default.pytorch")
            ),
            "depth_model": self.app.get_resource(
                os.path.join("depth_models", "model_city2kitti.meta")
            ),
            "yolo_weights": self.app.get_resource(
                os.path.join("yolo", "yolov3.weights")
            ),
            "yolo_v": self.app.get_resource(os.path.join("yolo", "yolov3.cfg")),
            "coco_names": self.app.get_resource(os.path.join("yolo", "coco.names")),
            "object_detection_dir": self.savePathJoin("ObjectDetection"),
            "plot_speed_dir": PLOT_SPEED_DIR,
            "plot_crash_dir": PLOT_CRASH_DIR,
            "numbers_dir": NP_DIR,
            "plot_error_dir": PLOT_ERROR_DIR,
            "speed_gt": self.user["GT"],
            "vid_path": self.user["Video"],
            "super_pixel_method": self.super_pixel_method,
            "super_pixel_dir": SUPER_PIXEL_DIR,
            "send_video_frame": False,
            "create_csv": self.ui.c_csv.isChecked(),
            "create_draw": self.ui.c_draw.isChecked(),
            "create_velocity": self.ui.c_velocity.isChecked(),
            "create_video_fps": int(self.ui.t_fps.text()),
            "optimize_params": self.ui.c_optimize.isChecked(),
            "super_pixel_label_dir": os.path.join(
                self.savePathJoin("Super_Pixel"), self.super_pixel_method
            ),
        }

    def startRun(self):
        """Check for errors and start calculations
        """
        self.checkFiles()
        if self.errorChecks():
            logging.info("Error Check Failed")
            return

        self.disableButtons()
        self.sendUser.emit(self.user)
        logging.info("Start Run Class")
        self.createDirs()
        self.buildRunDict()

    def startCalcThread(self):
        """Starting calculations on another thread
        """
        # 1 - create Worker and Thread inside the Form
        self.worker = calcRunner.CalculationRunner(self.params_dict)
        self.thread = QThread()  # no parent!

        self.worker.labelUpdate.connect(self.labelUpdate)

        self.worker.update.connect(self.progressUpdate)

        self.worker.error.connect(self.calculateError)

        self.worker.finished.connect(self.finishThread)

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.startThread)
        # 6 - Start the thread
        self.thread.start()

    def progressUpdate(self, value):
        """Updating both progressbar
        
        Arguments:
            value {int} -- current progress
        """
        self.progressBar.setValue(value)
        logging.info("Update progressbar to: {0}".format(self.run_count))
        if self.progressAllBar is not None:
            self.run_count += 1
            self.progressAllBar.setValue(self.run_count)

    def finishThread(self):
        """Clean up after calculations thread finished
        """
        logging.info("Fin Thread")
        self.buildCreatedDict()
        self.cleanThread()
        self.accept()

    def cleanThread(self):
        """Cleans the active thread
        """
        logging.info("Clean Thread")
        self.thread.quit()
        self.thread.wait()

    def labelUpdate(self, run_dict):
        """Update progressbar label's text to show the current calculation
        
        Arguments:
            run_dict {dictionary} -- dictionary of the current process
        """
        self.progressBar.reset()
        self.progressBar.setMinimum(1)
        self.progressBar.setMaximum(run_dict["Progress"])
        self.progressLabel.setText(run_dict["Text"])

    def showProgressBar(self):
        """Creates two progressbars to show how the calculation progresses
        """
        logging.info("Show progress bar")
        self.progressLabel = QLabel(self)
        font = QFont()
        font.setFamily("GE Inspira")
        font.setPointSize(20)
        self.progressLabel.setFont(font)
        self.progressLabel.setAlignment(Qt.AlignCenter)
        self.progressLabel.setText("Hello")
        self.ui.layout_v.addWidget(self.progressLabel)

        self.progressBar = QProgressBar(self)  # Progress bar created
        self.progressBar.setRange(0, 0)
        self.ui.layout_v.addWidget(self.progressBar)

    def addAllProgressBar(self):
        """Adds the progress bar which tracks the progress of all calculations
        """
        all_run = sum(
            [
                self.run_dict[key]["Progress"]
                for key in self.run_dict
                if self.run_dict[key]["Run"]
            ]
        )
        logging.info("All run: {0}".format(all_run))
        self.progressAllBar = QProgressBar(self)  # Progress bar created
        self.progressAllBar.setMinimum(1)
        self.progressAllBar.setMaximum(all_run)
        self.ui.layout_v.addWidget(self.progressAllBar)
        self.progressAllBar.setValue(1)

    def buildCreatedDict(self):
        """Build dictionary containing the created plots (if there's any).
        Send signal with this dictionary
        """
        self.created = {}
        if self.ui.c_speed_plot.isChecked():
            self.created["Speed_Plot"] = self.savePathJoin(PLOT_SPEED_DIR)
        if self.ui.c_error_plot.isChecked() and self.no_error:
            self.created["Error_Plot"] = self.savePathJoin(PLOT_ERROR_DIR)
        if self.ui.c_crash_plot.isChecked():
            self.created["Crash_Plot"] = self.savePathJoin(PLOT_CRASH_DIR)
        self.sendCreated.emit(self.created)

    def buildRunDict(self):
        """Create images from video in another thread, loads the image number otherwise
        """
        self.showProgressBar()
        ori_images = 0
        if self.img_exist:
            ori_images = len(listDirectory(self.savePathJoin("Images")))
            self.buildRunDictMain(ori_images)
        else:
            self.run_dict["Video"] = {
                "Run": True,
                "Progress": ori_images,
                "Text": "Preparing video",
            }
            self.buildParamsDict()
            self.params_dict["send_video_frame"] = True

            self.progressLabel.setText("Create images from video")

            self.worker = calcRunner.CalculationRunner(self.params_dict)  # no parent!
            self.thread = QThread()  # no parent!

            self.worker.labelUpdate.connect(self.labelUpdate)

            self.worker.update.connect(self.progressUpdate)
            self.worker.videoFrame.connect(self.setVidFrame)

            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.startThread)
            self.thread.start()

    def setVidFrame(self, ori_images):
        """Gets called when the image creation from video finishes
        
        Arguments:
            ori_images {int} -- number of images in the video
        """
        self.cleanThread()
        if ori_images == 0:
            logging.critical("Video Image number 0")
        else:
            self.buildRunDictMain(ori_images)

    def buildRunDictMain(self, ori_images):
        """Build dictionary with all calculations
        Dictionary fields:
            -Run: {bool} true if needs to be calculated, false otherwise
            -Progress: {int} the amount of steps to complete the calculation (used to update the progressbar)
            -Text: {str} the text to be showed in the progressbar label's when the calculation is running   
        """
        self.run_dict["Of"] = {
            "Run": not self.of_exist,
            "Progress": ori_images,
            "Text": "Running optical flow",
        }
        self.run_dict["Back_Of"] = {
            "Run": not self.back_of_exist,
            "Progress": ori_images,
            "Text": "Running back optical flow",
        }
        self.run_dict["Depth"] = {
            "Run": not self.depth_exist,
            "Progress": ori_images,
            "Text": "Running depth estimation",
        }
        self.run_dict["Speed"] = {
            "Run": True,
            "Progress": ori_images,
            "Text": "Running speed estimation",
        }
        self.run_dict["Optimization"] = {
            "Run": self.ui.c_optimize.isChecked(),
            "Progress": ori_images * 9,
            "Text": "Running parameter optimization",
        }

        self.run_dict["Of_Vid"] = {
            "Run": self.ui.c_of.isChecked(),
            "Progress": ori_images,
            "Text": "Creating optical flow video",
        }
        self.run_dict["Back_Of_Vid"] = {
            "Run": self.ui.c_back_of.isChecked(),
            "Progress": ori_images,
            "Text": "Creating backward optical flow video",
        }
        self.run_dict["Depth_Vid"] = {
            "Run": self.ui.c_depth.isChecked(),
            "Progress": ori_images,
            "Text": "Creating depth estimation video",
        }

        self.run_dict["Speed_Plot"] = {
            "Run": self.ui.c_speed_plot.isChecked(),
            "Progress": ori_images,
            "Text": "Creating plot for speed values",
        }
        self.run_dict["Crash_Plot"] = {
            "Run": self.ui.c_crash_plot.isChecked(),
            "Progress": ori_images,
            "Text": "Creating plot for time to crash",
        }
        self.run_dict["Error_Plot"] = {
            "Run": self.ui.c_error_plot.isChecked() and self.gt_exist,
            "Progress": ori_images,
            "Text": "Creating plot for speed error",
        }

        self.run_dict["Speed_Plot_Video"] = {
            "Run": self.ui.c_speed_plot_video.isChecked(),
            "Progress": ori_images,
            "Text": "Creating speed plot video",
        }
        self.run_dict["Error_Plot_Video"] = {
            "Run": self.ui.c_error_plot_video.isChecked() and self.gt_exist,
            "Progress": ori_images,
            "Text": "Creating error plot video",
        }
        self.run_dict["Crash_Plot_Video"] = {
            "Run": self.ui.c_crash_plot_video.isChecked(),
            "Progress": ori_images,
            "Text": "Creating time to crash plot video",
        }

        self.run_dict["Super_Pixel_Video"] = {
            "Run": self.ui.combo_superpixel.currentIndex() != 0
            and self.ui.c_super_pixel_video.isChecked(),
            "Progress": ori_images,
            "Text": "Creating super pixel video",
        }
        self.run_dict["Super_Pixel_Label"] = {
            "Run": self.create_super_pixel_label,
            "Progress": ori_images,
            "Text": "Creating {0} superpixel labels".format(self.super_pixel_method),
        }

        self.run_dict["Object_Detection"] = {
            "Run": (
                self.ui.c_object_detection.isChecked()
                or self.ui.c_crash_plot.isChecked()
            )
            and not self.object_detection_dir_exist,
            "Progress": ori_images,
            "Text": "Running Object Detection",
        }

        self.addAllProgressBar()
        self.buildParamsDict()
        self.saveUser()
        self.startCalcThread()

    def disableButtons(self):
        """Disable widgets while the calculation is running
        """
        self.ui.b_run.setEnabled(False)
        self.ui.b_colour.setEnabled(False)
        self.ui.b_ground_truth.setEnabled(False)
        self.ui.b_vid.setEnabled(False)
        self.ui.b_save.setEnabled(False)
        self.ui.t_low.setEnabled(False)
        self.ui.t_high.setEnabled(False)
        self.ui.t_fps.setEnabled(False)
        self.ui.combo_superpixel.setEnabled(False)
        self.ui.c_super_pixel_video.setEnabled(False)
        self.ui.c_csv.setEnabled(False)
        self.ui.c_draw.setEnabled(False)
        self.ui.c_velocity.setEnabled(False)
        self.ui.c_of.setEnabled(False)
        self.ui.c_back_of.setEnabled(False)
        self.ui.c_depth.setEnabled(False)
        self.ui.c_speed_plot.setEnabled(False)
        self.ui.c_error_plot.setEnabled(False)
        self.ui.c_crash_plot.setEnabled(False)
        self.ui.c_error_plot_video.setEnabled(False)
        self.ui.c_speed_plot_video.setEnabled(False)
        self.ui.c_crash_plot_video.setEnabled(False)
        self.ui.c_optimize.setEnabled(False)
        self.ui.c_object_detection.setEnabled(False)

    def savePathJoin(self, path):
        """Join the given path with the save path (stored in user info)
        
        Arguments:
            path {str} -- path to join after the save path
        
        Returns:
            str -- joined path
        """
        return os.path.join(self.user["Save"], path)

    def createDir(self, dir_name):
        """Create directory with the given name in the save path (stored in user info)
        
        Arguments:
            dir_name {str} -- path to the newly created dir
        """
        os.mkdir(os.path.join(self.user["Save"], dir_name))

    def createDirs(self):
        """Create or recreate (destroy and create) directories for the calculations
        """
        logging.info("Creating Directories")

        if not self.img_exist:
            self.reCreateDir(self.savePathJoin("Images"))
        if not self.of_exist:
            self.reCreateDir(self.savePathJoin("Of"))
        if not self.back_of_exist:
            self.reCreateDir(self.savePathJoin("Back_Of"))
        if not self.depth_exist:
            self.reCreateDir(self.savePathJoin("Depth"))
        if not self.object_detection_dir_exist and (
            self.ui.c_object_detection.isChecked() or self.ui.c_crash_plot.isChecked()
        ):
            self.reCreateDir(self.savePathJoin("ObjectDetection"))
        if self.super_pixel_method != "" and not os.path.exists(
            os.path.join(self.savePathJoin("Super_Pixel"), self.super_pixel_method)
        ):
            os.makedirs(
                os.path.join(self.savePathJoin("Super_Pixel"), self.super_pixel_method)
            )

        self.reCreateDir(RESULTS)
        self.reCreateDir(NP_DIR)
        self.reCreateDir(MASK_DIR)

        if self.ui.c_crash_plot.isChecked():
            self.reCreateDir(PLOT_CRASH_DIR)
        if self.ui.c_draw.isChecked():
            self.reCreateDir(DRAW_DIR)
        if self.ui.c_velocity.isChecked():
            self.reCreateDir(VL_DIR)
        if self.ui.c_speed_plot.isChecked():
            self.reCreateDir(PLOT_SPEED_DIR)
        if self.super_pixel_method != "":
            self.reCreateDir(SUPER_PIXEL_DIR)
        if self.user["GT"] != "" and self.ui.c_error_plot.isChecked():
            self.reCreateDir(PLOT_ERROR_DIR)

    def reCreateDir(self, name):
        """Create directory with the given name in save dir (stored in user info).
        If it already exists, then delete it first
        
        Arguments:
            name {str} -- name of the new directory
        """
        path = self.savePathJoin(name)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def pickColour(self):
        """Opens a QColorDialog to choose a colour
        """
        colour = QColorDialog.getColor()
        if colour.isValid():
            self.user["Colour"] = colour.name()
            self.ui.l_colour.setText(self.user["Colour"])
