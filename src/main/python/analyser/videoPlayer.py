from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, QObject

class VideoPlayer(QtWidgets.QLabel):
    """QLabel with a custom resizeEvent
    
    Arguments:
        QtWidgets {QLabel}
    """
    resizeSignal = QtCore.pyqtSignal(int, int)
    def __init__(self):
        super().__init__()  
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setText("")
        self.setMinimumSize(1,1)

    def resizeEvent(self, event):
        self.resizeSignal.emit(self.width(), self.height())