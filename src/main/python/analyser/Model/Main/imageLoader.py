from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class WorkerSignals(QObject):
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()          

    @pyqtSlot()
    def run(self):
        """Executes the function (fn) which was given in the constructor and emits a signal with the result
        """
        result = self.fn(*self.args, **self.kwargs)
        self.signals.result.emit(result)  # Return the result of the processing
    