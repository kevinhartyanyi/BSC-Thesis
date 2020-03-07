# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialog.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(395, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.b_save = QtWidgets.QPushButton(Dialog)
        self.b_save.setObjectName("b_save")
        self.horizontalLayout_2.addWidget(self.b_save)
        self.l_save = QtWidgets.QLabel(Dialog)
        self.l_save.setText("")
        self.l_save.setObjectName("l_save")
        self.horizontalLayout_2.addWidget(self.l_save)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.b_vid = QtWidgets.QPushButton(Dialog)
        self.b_vid.setObjectName("b_vid")
        self.horizontalLayout.addWidget(self.b_vid)
        self.l_vid = QtWidgets.QLabel(Dialog)
        self.l_vid.setText("")
        self.l_vid.setObjectName("l_vid")
        self.horizontalLayout.addWidget(self.l_vid)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.b_of = QtWidgets.QPushButton(Dialog)
        self.b_of.setObjectName("b_of")
        self.horizontalLayout_3.addWidget(self.b_of)
        self.l_of = QtWidgets.QLabel(Dialog)
        self.l_of.setText("")
        self.l_of.setObjectName("l_of")
        self.horizontalLayout_3.addWidget(self.l_of)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.b_depth = QtWidgets.QPushButton(Dialog)
        self.b_depth.setObjectName("b_depth")
        self.horizontalLayout_4.addWidget(self.b_depth)
        self.l_depth = QtWidgets.QLabel(Dialog)
        self.l_depth.setText("")
        self.l_depth.setObjectName("l_depth")
        self.horizontalLayout_4.addWidget(self.l_depth)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.b_run = QtWidgets.QPushButton(Dialog)
        self.b_run.setObjectName("b_run")
        self.verticalLayout.addWidget(self.b_run)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.b_save.setText(_translate("Dialog", "Save Path"))
        self.b_vid.setText(_translate("Dialog", "Open Video"))
        self.b_of.setText(_translate("Dialog", "Optical flow"))
        self.b_depth.setText(_translate("Dialog", "Depth"))
        self.b_run.setText(_translate("Dialog", "Run"))
