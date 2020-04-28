from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject
import time


class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal()

    def __init__(self, current_idx, end_idx, fps):
        super(Worker, self).__init__()
        self.current_idx = current_idx
        self.end_idx = end_idx
        self.wait = 1 / fps
        self.running = True

    @pyqtSlot()
    def startCounting(self):  # A slot takes no params
        """
        Start counting by fps and return signal to update video
        """
        while self.current_idx < self.end_idx - 1 and self.running:
            self.intReady.emit()
            time.sleep(self.wait)
            self.current_idx += 1

        self.finished.emit()

    def stop(self):
        """
        Stops the counting
        """
        self.running = False
