# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GammaWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 630)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(10, 0, 381, 621))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(140, 10, 101, 16))
        self.label_2.setObjectName("label_2")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(30, 30, 101, 16))
        self.label.setObjectName("label")
        self.lblPreGamma = QtWidgets.QLabel(self.groupBox)
        self.lblPreGamma.setGeometry(QtCore.QRect(30, 70, 331, 211))
        self.lblPreGamma.setText("")
        self.lblPreGamma.setObjectName("lblPreGamma")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(30, 300, 71, 16))
        self.label_4.setObjectName("label_4")
        self.lblPostGamma = QtWidgets.QLabel(self.groupBox)
        self.lblPostGamma.setGeometry(QtCore.QRect(30, 320, 311, 231))
        self.lblPostGamma.setText("")
        self.lblPostGamma.setObjectName("lblPostGamma")
        self.slider_gamma = QtWidgets.QSlider(self.groupBox)
        self.slider_gamma.setGeometry(QtCore.QRect(22, 560, 301, 22))
        self.slider_gamma.setMinimum(1)
        self.slider_gamma.setMaximum(20)
        self.slider_gamma.setProperty("value", 10)
        self.slider_gamma.setOrientation(QtCore.Qt.Horizontal)
        self.slider_gamma.setObjectName("slider_gamma")
        self.pbValidateGamma = QtWidgets.QPushButton(self.groupBox)
        self.pbValidateGamma.setGeometry(QtCore.QRect(20, 580, 93, 28))
        self.pbValidateGamma.setObjectName("pbValidateGamma")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_2.setText(_translate("Form", "Gamma Preview"))
        self.label.setText(_translate("Form", "Original Image"))
        self.label_4.setText(_translate("Form", "Final Image"))
        self.pbValidateGamma.setText(_translate("Form", "Validate"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
