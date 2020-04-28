from View.Run.dialog import Dialog
from main import MainWindow
from fbs_runtime.application_context.PyQt5 import ApplicationContext
import qdarkgraystyle
import unittest
from PyQt5.QtTest import QTest
import os
from Model.Algorithms.utils import listDirectory
import time
import shutil
import Model.Run.calcRunner as calcRunner


appctxt = ApplicationContext()
appctxt.app.setStyleSheet(qdarkgraystyle.load_stylesheet())


class RunTester(unittest.TestCase):
    def setUp(self):
        self.run = Dialog(app=appctxt)

    def test_defaults(self):
        self.assertEqual(self.run.ui.combo_superpixel.currentIndex(), 0)
        self.assertEqual(self.run.ui.c_csv.isEnabled(), False)
        self.assertEqual(self.run.ui.c_draw.isChecked(), False)
        self.assertEqual(self.run.ui.c_velocity.isChecked(), False)
        self.assertEqual(self.run.ui.t_low.text(), "0.0")
        self.assertEqual(self.run.ui.t_high.text(), "1.0")
        self.assertEqual(self.run.ui.t_fps.text(), "30")
        self.assertEqual(self.run.ui.c_of.isChecked(), False)
        self.assertEqual(self.run.ui.c_depth.isChecked(), False)
        self.assertEqual(self.run.ui.c_back_of.isChecked(), False)
        self.assertEqual(self.run.ui.c_error_plot_video.isEnabled(), False)
        self.assertEqual(self.run.ui.c_speed_plot_video.isEnabled(), False)
        self.assertEqual(self.run.ui.c_error_plot.isEnabled(), self.run.gt_exist)
        self.assertEqual(self.run.ui.c_speed_plot.isChecked(), False)

    def test_fps_change(self):
        self.assertEqual(self.run.fps, 30)
        self.assertEqual(self.run.ui.t_fps.text(), "30")
        self.run.ui.t_fps.setText("20")
        self.assertEqual(self.run.fps, 20)

        self.run.ui.t_fps.setText("-1")
        self.assertEqual(self.run.fps, 1)

        self.run.ui.t_fps.setText("100")
        self.assertEqual(self.run.fps, self.run.fps_limit)

        self.run.ui.t_fps.setText("asg")
        self.assertEqual(self.run.fps, 30)

    def test_low_change(self):
        self.run.ui.t_low.setText("-1")
        self.assertEqual(self.run.low, 0.0)

        self.run.ui.t_low.setText("2.2")
        self.assertEqual(self.run.low, 0.0)

        self.run.ui.t_low.setText("ttt")
        self.assertEqual(self.run.low, 0.0)

    def test_high_change(self):
        self.run.ui.t_high.setText("-1")
        self.assertEqual(self.run.high, 1.0)

        self.run.ui.t_high.setText("2.2")
        self.assertEqual(self.run.high, 1.0)

        self.run.ui.t_high.setText("ttt")
        self.assertEqual(self.run.high, 1.0)

    def test_run_requirements(self):
        self.assertEqual(self.run.ui.b_run.isEnabled(), self.run.runRequirements())

    def test_speed_plot(self):
        self.assertEqual(self.run.ui.c_speed_plot_video.isEnabled(), False)
        self.run.ui.c_speed_plot.setChecked(True)
        self.assertEqual(self.run.ui.c_speed_plot_video.isEnabled(), True)

    def test_error_plot(self):
        self.assertEqual(self.run.ui.c_error_plot_video.isEnabled(), False)
        self.assertEqual(self.run.ui.c_error_plot.isEnabled(), self.run.gt_exist)
        self.run.user["GT"] = "something.npy"
        self.run.checkFiles()
        self.assertEqual(self.run.ui.c_error_plot_video.isEnabled(), False)
        self.assertEqual(self.run.ui.c_error_plot.isEnabled(), True)
        self.run.ui.c_error_plot.setChecked(True)
        self.assertEqual(self.run.ui.c_error_plot_video.isEnabled(), True)

    def test_super_pixel(self):
        self.assertEqual(self.run.ui.c_super_pixel_video.isEnabled(), False)
        self.run.ui.combo_superpixel.setCurrentIndex(1)
        self.run.checkFiles()
        self.assertEqual(self.run.ui.c_super_pixel_video.isEnabled(), True)

    def test_optimize(self):
        self.assertEqual(self.run.ui.t_low.isEnabled(), True)
        self.assertEqual(self.run.ui.t_high.isEnabled(), True)
        self.assertEqual(self.run.ui.c_optimize.isChecked(), False)
        self.run.ui.c_optimize.setChecked(True)
        self.assertEqual(self.run.ui.t_low.isEnabled(), False)
        self.assertEqual(self.run.ui.t_high.isEnabled(), False)

    def test_optimize_enable(self):
        self.assertEqual(self.run.ui.c_optimize.isEnabled(), self.run.gt_exist)
        self.run.user["GT"] = "something.npy"
        self.run.checkFiles()
        self.assertEqual(self.run.ui.c_optimize.isEnabled(), True)

    def test_checkFiles(self):
        self.test_defaults()
        self.run.user["Save"] = "/test"
        self.run.user["Of"] = "/tes2"
        self.run.user["GT"] = "/ground_truth"
        self.run.ui.c_optimize.setChecked(True)
        self.run.checkFiles()
        self.assertEqual(self.run.gt_exist, True)
        self.assertEqual(self.run.ui.t_low.isEnabled(), False)
        self.assertEqual(self.run.ui.t_high.isEnabled(), False)
        self.assertEqual(self.run.ui.c_optimize.isEnabled(), True)
        self.assertEqual(self.run.ui.c_error_plot.isEnabled(), True)

    def test_createDirs_run(self):
        self.run.user["Save"] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test"
        )

        img = os.path.join(self.run.user["Save"], "Images")
        depth = os.path.join(self.run.user["Save"], "Depth")
        of = os.path.join(self.run.user["Save"], "Of")
        back_of = os.path.join(self.run.user["Save"], "Back_Of")
        result = os.path.join(self.run.user["Save"], "results")

        numbers = os.path.join(result, "numbers")
        mask = os.path.join(result, "mask")

        self.assertEqual(
            os.path.exists(os.path.join(self.run.user["Save"], "Images")), True
        )
        self.run.checkFiles()
        self.run.createDirs()

        self.assertEqual(os.path.exists(depth), True)
        self.assertEqual(os.path.exists(of), True)
        self.assertEqual(os.path.exists(back_of), True)
        self.assertEqual(os.path.exists(result), True)
        self.assertEqual(os.path.exists(numbers), True)
        self.assertEqual(os.path.exists(mask), True)

    def test_plot_dir(self):

        self.run.user["Save"] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test"
        )

        img = os.path.join(self.run.user["Save"], "Images")

        self.run.ui.c_speed_plot.setChecked(True)
        self.run.ui.c_error_plot.setChecked(True)

        result = os.path.join(self.run.user["Save"], "results")
        plot_speed = os.path.join(result, "plot_speed")

        self.run.checkFiles()
        self.run.disableButtons()
        self.run.createDirs()

        self.assertEqual(os.path.exists(plot_speed), True)

    def prepare_run(self, empty=False):
        self.run.user["Save"] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test"
        )
        self.img = os.path.join(self.run.user["Save"], "Images")
        self.depth = os.path.join(self.run.user["Save"], "Depth")
        self.of = os.path.join(self.run.user["Save"], "Of")
        self.back_of = os.path.join(self.run.user["Save"], "Back_Of")
        self.result = os.path.join(self.run.user["Save"], "results")
        self.numbers = os.path.join(self.result, "numbers")
        self.mask = os.path.join(self.result, "mask")

        if empty:
            tmp = [self.depth, self.of, self.back_of, self.result]
            for t in tmp:
                if os.path.exists(t):
                    shutil.rmtree(t)

        self.run.checkFiles()
        self.run.disableButtons()
        self.run.createDirs()

        ori_images = len(listDirectory(self.img))
        self.run.run_dict["Of"] = {
            "Run": not self.run.of_exist,
            "Progress": ori_images,
            "Text": "Running optical flow",
        }
        self.run.run_dict["Back_Of"] = {
            "Run": not self.run.back_of_exist,
            "Progress": ori_images,
            "Text": "Running back optical flow",
        }
        self.run.run_dict["Depth"] = {
            "Run": not self.run.depth_exist,
            "Progress": ori_images,
            "Text": "Running depth estimation",
        }
        self.run.run_dict["Speed"] = {
            "Run": True,
            "Progress": ori_images,
            "Text": "Running speed estimation",
        }
        self.run.run_dict["Optimization"] = {
            "Run": self.run.ui.c_optimize.isChecked(),
            "Progress": ori_images * 9,
            "Text": "Running parameter optimization",
        }

        self.run.run_dict["Of_Vid"] = {
            "Run": self.run.ui.c_of.isChecked(),
            "Progress": ori_images,
            "Text": "Creating optical flow video",
        }
        self.run.run_dict["Back_Of_Vid"] = {
            "Run": self.run.ui.c_back_of.isChecked(),
            "Progress": ori_images,
            "Text": "Creating backward optical flow video",
        }
        self.run.run_dict["Depth_Vid"] = {
            "Run": self.run.ui.c_depth.isChecked(),
            "Progress": ori_images,
            "Text": "Creating depth estimation video",
        }

        self.run.run_dict["Speed_Plot"] = {
            "Run": self.run.ui.c_speed_plot.isChecked(),
            "Progress": ori_images,
            "Text": "Creating plot for speed values",
        }
        self.run.run_dict["Crash_Plot"] = {
            "Run": self.run.ui.c_crash_plot.isChecked(),
            "Progress": ori_images,
            "Text": "Creating plot for time to crash",
        }
        self.run.run_dict["Error_Plot"] = {
            "Run": self.run.ui.c_error_plot.isChecked() and self.run.gt_exist,
            "Progress": ori_images,
            "Text": "Creating plot for speed error",
        }

        self.run.run_dict["Speed_Plot_Video"] = {
            "Run": self.run.ui.c_speed_plot_video.isChecked(),
            "Progress": ori_images,
            "Text": "Creating speed plot video",
        }
        self.run.run_dict["Error_Plot_Video"] = {
            "Run": self.run.ui.c_error_plot_video.isChecked() and self.run.gt_exist,
            "Progress": ori_images,
            "Text": "Creating error plot video",
        }
        self.run.run_dict["Crash_Plot_Video"] = {
            "Run": self.run.ui.c_crash_plot_video.isChecked(),
            "Progress": ori_images,
            "Text": "Creating time to crash plot video",
        }

        self.run.run_dict["Super_Pixel_Video"] = {
            "Run": self.run.ui.combo_superpixel.currentIndex() != 0
            and self.run.ui.c_super_pixel_video.isChecked(),
            "Progress": ori_images,
            "Text": "Creating super pixel video",
        }
        self.run.run_dict["Super_Pixel_Label"] = {
            "Run": self.run.create_super_pixel_label,
            "Progress": ori_images,
            "Text": "Creating {0} superpixel labels".format(
                self.run.super_pixel_method
            ),
        }

        self.run.run_dict["Object_Detection"] = {
            "Run": (
                self.run.ui.c_object_detection.isChecked()
                or self.run.ui.c_crash_plot.isChecked()
            )
            and not self.run.object_detection_dir_exist,
            "Progress": ori_images,
            "Text": "Running Object Detection",
        }

        self.run.addAllProgressBar()
        self.run.buildParamsDict()
        runner = calcRunner.CalculationRunner(self.run.params_dict)
        runner.startThread()

    def test_simple_run(self):
        self.prepare_run(empty=True)
        img_num = len(listDirectory(self.img, extension=".png"))
        depth_num = len(listDirectory(self.depth, extension=".png"))
        of_num = len(listDirectory(self.of, extension=".png"))
        back_of_num = len(listDirectory(self.back_of, extension=".png"))

        mask_num = len(listDirectory(self.mask, extension=".png"))

        self.assertEqual(img_num, depth_num)
        self.assertEqual(of_num, back_of_num)
        self.assertEqual(img_num - 1, of_num)

        self.assertEqual(img_num - 1, mask_num)

    def test_plot_run(self):
        self.run.ui.c_speed_plot.setChecked(True)
        self.prepare_run(empty=True)
        plot_speed = os.path.join(self.result, "plot_speed")
        self.assertEqual(os.path.exists(plot_speed), True)

        img_num = len(listDirectory(self.img, extension=".png"))
        speed_num = len(listDirectory(plot_speed, extension=".png"))
        self.assertEqual(img_num - 1, speed_num)

    def test_superpixel_run(self):
        self.run.ui.combo_superpixel.setCurrentIndex(1)
        self.run.changeSuperPixelMethod(1)
        self.prepare_run()
        superpixel_dir = os.path.join(self.result, "super_pixel")

        self.assertEqual(os.path.exists(superpixel_dir), True)

        img_num = len(listDirectory(self.img, extension=".png"))
        superpixel_num = len(listDirectory(superpixel_dir, extension=".png"))
        self.assertEqual(img_num - 1, superpixel_num)


class MainTester(unittest.TestCase):
    def setUp(self):
        self.run = MainWindow(app=appctxt)

    def test_defaults(self):
        self.assertEqual(self.run.ui.b_video_right.isEnabled(), False)
        self.assertEqual(self.run.ui.b_video_left.isEnabled(), False)
        self.assertEqual(self.run.ui.b_video_up.isEnabled(), False)
        self.assertEqual(self.run.ui.b_video_down.isEnabled(), False)
        self.assertEqual(self.run.ui.b_plot_left.isEnabled(), False)
        self.assertEqual(self.run.ui.b_plot_right.isEnabled(), False)
        self.assertEqual(self.run.ui.actionPlay.isEnabled(), False)
        self.assertEqual(self.run.ui.actionOF.isEnabled(), False)
        self.assertEqual(self.run.ui.actionDepth.isEnabled(), False)
        self.assertEqual(self.run.ui.actionOriginal.isEnabled(), False)
        self.assertEqual(self.run.ui.actionOFDirections.isEnabled(), False)
        self.assertEqual(self.run.ui.actionOFArrows.isEnabled(), False)
        self.assertEqual(self.run.ui.actionSuperPixel.isEnabled(), False)
        self.assertEqual(self.run.ui.actionMask.isEnabled(), False)
        self.assertEqual(self.run.ui.actionBackOF.isEnabled(), False)
        self.assertEqual(self.run.ui.b_rerun.isEnabled(), True)

    def test_fps_change(self):
        self.assertEqual(self.run.image_holder.fps, 30)
        self.assertEqual(self.run.ui.t_fps.text(), "30")
        self.run.ui.t_fps.setText("20")
        self.assertEqual(self.run.image_holder.fps, 20)

        self.run.ui.t_fps.setText("-1")
        self.assertEqual(self.run.image_holder.fps, 1)

        self.run.ui.t_fps.setText("100")
        self.assertEqual(self.run.image_holder.fps, self.run.fps_limit)

        self.run.ui.t_fps.setText("asg")
        self.assertEqual(self.run.image_holder.fps, 30)

    def test_description(self):
        self.assertEqual(self.run.ui.l_description.text(), "")
        self.run.cycle_vid.add("original", "/path")
        self.run.cycle_vid.add("of", "/path")
        self.run.changeDescription()
        self.assertEqual(self.run.ui.l_description.text(), "The original video")
        self.run.cycle_vid.up()
        self.run.changeDescription()
        self.assertEqual(
            self.run.ui.l_description.text(),
            "Optical flow (motion of image objects between two consecutive frames)",
        )

    def test_cycle(self):
        self.run.cycle_vid.add("original", "/path")
        self.run.cycle_vid.add("of", "/path")
        self.run.cycle_vid.add("back_of", "/path")
        self.run.cycle_vid.add("depth", "/path")
        self.run.cycle_vid.add("velocity", "/path")
        self.run.cycle_vid.add("mask", "/path")
        self.run.cycle_vid.add("draw", "/path")
        self.run.cycle_vid.add("super_pixel", "/path")
        self.assertEqual(self.run.cycle_vid.currentType(), "original")
        self.run.cycle_vid.up()
        self.assertEqual(self.run.cycle_vid.currentType(), "of")
        self.run.cycle_vid.up()
        self.assertEqual(self.run.cycle_vid.currentType(), "back_of")
        self.run.cycle_vid.down()
        self.assertEqual(self.run.cycle_vid.currentType(), "of")
        self.run.cycle_vid.up()
        self.run.cycle_vid.up()
        self.assertEqual(self.run.cycle_vid.currentType(), "depth")
        self.run.cycle_vid.up()
        self.assertEqual(self.run.cycle_vid.currentType(), "velocity")
        self.run.cycle_vid.up()
        self.assertEqual(self.run.cycle_vid.currentType(), "mask")
        self.run.cycle_vid.up()
        self.run.cycle_vid.up()
        self.run.cycle_vid.down()
        self.assertEqual(self.run.cycle_vid.currentType(), "draw")
        self.run.cycle_vid.up()
        self.assertEqual(self.run.cycle_vid.currentType(), "super_pixel")
        self.run.cycle_vid.up()
        self.assertEqual(self.run.cycle_vid.currentType(), "original")


if __name__ == "__main__":
    unittest.main()
