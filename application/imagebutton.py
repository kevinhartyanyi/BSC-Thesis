from PyQt5 import QtCore, QtGui, QtWidgets


class ImageButton(QtWidgets.QAbstractButton):
    def __init__(self, parent=None):
        super(ImageButton, self).__init__(parent)
        self.pixmap = None
        self.pixmap_hover = None
        self.pixmap_pressed = None

        self.pressed.connect(self.update)
        self.released.connect(self.update)
    
    def setUpPixmaps(self, pixmap, pixmap_hover, pixmap_pressed):
        self.pixmap = pixmap
        self.pixmap_hover = pixmap_hover
        self.pixmap_pressed = pixmap_pressed

    def paintEvent(self, event):
        pix = self.pixmap_hover if self.underMouse() else self.pixmap
        if self.isDown():
            pix = self.pixmap_pressed

        painter = QtGui.QPainter(self)
        painter.drawPixmap(event.rect(), pix)

    def enterEvent(self, event):
        self.update()

    def leaveEvent(self, event):
        self.update()

    def sizeHint(self):
        return QtGui.QSize(200, 200)