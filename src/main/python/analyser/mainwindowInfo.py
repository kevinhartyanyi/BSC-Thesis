from PyQt5.QtWidgets import QDialog
from mainwindowInfoUI import Ui_Dialog


class MainWindowInfo(QDialog, Ui_Dialog):

    def __init__(self, app, parent=None):
        super(MainWindowInfo, self).__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle("Information")
        
        self.app = app

        self.signalSetup()
    
    def signalSetup(self):
        """
        Setup for signal connections
        """
        pass
