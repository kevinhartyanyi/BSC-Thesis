from PyQt5.QtWidgets import QDialog
from View.Main.loggerUI import Ui_Dialog
from PyQt5.QtWidgets import QPlainTextEdit
from PyQt5.QtCore import QObject, pyqtSignal
import logging

class LoggerText(logging.Handler, QObject):
    appendPlainText = pyqtSignal(str)

    def __init__(self, parent):
        super(LoggerText, self).__init__()
        QObject.__init__(self)

        self.widget = QPlainTextEdit(parent)
        self.widget.setReadOnly(True)
        self.appendPlainText.connect(self.widget.appendPlainText)

    def emit(self, record):
        msg = self.format(record)
        self.appendPlainText.emit(msg)


class LogInfo(QDialog, Ui_Dialog):

    def __init__(self, parent=None):
        super(LogInfo, self).__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle("Log Information")
        log_handler = LoggerText(self)
        log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(module)s %(funcName)s %(message)s"))
        logging.getLogger().addHandler(log_handler)
        logging.getLogger().setLevel(logging.DEBUG)
        self.ui.layout.insertWidget(1, log_handler.widget)

        fh = logging.FileHandler("log_info.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(module)s %(funcName)s %(message)s"))
        logging.getLogger().addHandler(fh)
    
