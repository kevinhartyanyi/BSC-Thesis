from dialog import Dialog
from main import MainWindow
from fbs_runtime.application_context.PyQt5 import ApplicationContext
import qdarkgraystyle
import unittest
from PyQt5.QtTest import QTest

appctxt = ApplicationContext()
appctxt.app.setStyleSheet(qdarkgraystyle.load_stylesheet())



class DialogTester(unittest.TestCase):
    def setUp(self):
        self.run = Dialog(app=appctxt)

    def test_defaults(self):
        self.assertEqual(self.run.ui.combo_superpixel.currentIndex(), 0)
        self.assertEqual(self.run.ui.c_csv.isEnabled(), False)
        self.assertEqual(self.run.ui.c_draw.isChecked(), False)
        self.assertEqual(self.run.ui.c_velocity.isChecked(), False)
        self.assertEqual(self.run.ui.t_low.text(), "0.309")
        self.assertEqual(self.run.ui.t_high.text(), "1.0")
        self.assertEqual(self.run.ui.t_fps.text(), "30")
        self.assertEqual(self.run.ui.c_of.isChecked(), False)
        self.assertEqual(self.run.ui.c_depth.isChecked(), False)
        self.assertEqual(self.run.ui.c_back_of.isChecked(), False)
        self.assertEqual(self.run.ui.c_error_plot_video.isEnabled(), False)
        self.assertEqual(self.run.ui.c_speed_plot_video.isEnabled(), False)
        self.assertEqual(self.run.ui.c_error_plot.isEnabled(), False)
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
        self.assertEqual(self.run.low, 0.309)

        self.run.ui.t_low.setText("2.2")
        self.assertEqual(self.run.low, 0.309)

        self.run.ui.t_low.setText("ttt")
        self.assertEqual(self.run.low, 0.309)
    
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
        self.assertEqual(self.run.ui.c_error_plot.isEnabled(), False)
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
        self.assertEqual(self.run.ui.l_description.text(), "Optical flow (motion of image objects between two consecutive frames)")

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
