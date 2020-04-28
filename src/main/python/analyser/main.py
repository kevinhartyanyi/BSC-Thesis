import qdarkgraystyle
from fbs_runtime.application_context.PyQt5 import ApplicationContext
import sys
from View.Main.mainwindow import MainWindow

if __name__ == "__main__":
    appctxt = ApplicationContext()
    appctxt.app.setStyleSheet(qdarkgraystyle.load_stylesheet())
    window = MainWindow(app=appctxt)
    window.show()
    exit_code = appctxt.app.exec_()
    sys.exit(exit_code)
