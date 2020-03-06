from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, QObject

class VideoPlayer(QtWidgets.QLabel):
    resizeSignal = QtCore.pyqtSignal(int, int)
    def __init__(self):
        super().__init__()  
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        self.setText("")

    def resizeEvent(self, event):
        self.resizeSignal.emit(self.width(), self.height())