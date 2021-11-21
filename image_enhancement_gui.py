#!/usr/bin/env python

# Created by Shahar Gino at November 2021
# All rights reserved

import sys
import numpy as np
import cv2, imutils
from os import environ
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPalette 
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # supress tensorflow messages 
from image_enhancement import image_enhance, image_enhance_defparams, iqa_score


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(536, 571)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # Top Layout (reshape):
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        
        # Parameters + Button + Spacer layout:
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        
        # Image ("layout") and Parameters Layout:
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        
        # Image ("label"):
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setObjectName("label")
        self.label.setBackgroundRole(QPalette.Base)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.label.setScaledContents(True)        
       
        # Scroll Area:
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.label)
        self.scrollArea.setVisible(False)
        
        self.verticalLayout_3.addWidget(self.scrollArea)

        # Parameters Layout:
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        self.defparams = image_enhance_defparams()

        self.parameter_widgets = []

        for param_name, param_value in self.defparams.items():
            
            layout_obj_name = '%s_layout' % param_name
            setattr(self, layout_obj_name, QtWidgets.QHBoxLayout())
            layout_obj = getattr(self, layout_obj_name)
            layout_obj.setObjectName(layout_obj_name)
            
            label_obj_name = '%s_label' % param_name
            setattr(self, label_obj_name, QtWidgets.QLabel())
            label_obj = getattr(self, label_obj_name)
            layout_obj.addWidget(label_obj)
            self.parameter_widgets.append(label_obj)

            textbox_obj_name = '%s_textbox' % param_name
            setattr(self, textbox_obj_name, QtWidgets.QLineEdit(self.centralwidget))
            textbox_obj = getattr(self, textbox_obj_name)
            textbox_obj.move(20, 20)
            textbox_obj.resize(280,40)
            textbox_obj.setText(str(param_value))
            layout_obj.addWidget(textbox_obj)
            self.parameter_widgets.append(textbox_obj)

            self.verticalLayout.addLayout(layout_obj)
        
        self.hide_params()

        self.verticalLayout_3.addLayout(self.verticalLayout)
        
        self.gridLayout.addLayout(self.verticalLayout_3, 0, 0, 1, 2)

        # Buttons Layout:
        self.verticalLayout_2 = QtWidgets.QHBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.hide()
        self.verticalLayout_2.addWidget(self.pushButton)
        
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_2.addWidget(self.pushButton_2)
        
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.hide()
        self.verticalLayout_2.addWidget(self.pushButton_3)
        
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.hide()
        self.verticalLayout_2.addWidget(self.pushButton_4)
        
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.hide()
        self.verticalLayout_2.addWidget(self.pushButton_5)
        
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout_2.addWidget(self.pushButton_6)
        
        self.gridLayout.addLayout(self.verticalLayout_2, 1, 0, 1, 1)
        
        # Spacer:
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        
        self.gridLayout.addItem(spacerItem, 1, 1, 1, 1)
       
        # Top epilog:
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        
        # Status-Bar:
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # Actions bind: 
        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(self.savePhoto)
        self.pushButton_2.clicked.connect(self.loadImage)
        self.pushButton_3.clicked.connect(self.launch)
        self.pushButton_4.clicked.connect(self.toggle_params)
        self.pushButton_5.clicked.connect(self.image_quality)
        self.pushButton_6.clicked.connect(self.quit)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        # Variables initialization:
        self.filename = None
        self.tmp_img = None
        self.params_en = False
        self.scaleFactor = 0.0
        self.printer = QPrinter()
        
        # Actions and Menus
        self.createActions(MainWindow)
        self.createMenus(MainWindow)
   
    # -----------------------------------------------------------------------------------------

    def launch(self):

        params = {}
        for param_name in self.defparams.keys():
            params[param_name] = getattr(self, "%s_textbox" % param_name).text()
            if params[param_name].isdigit():
                params[param_name] = int(params[param_name])
            elif params[param_name].replace('.','',1).isdigit():
                params[param_name] = float(params[param_name])

        res_img = image_enhance(self.image, params)

        self.setPhoto(res_img)
    
    # -----------------------------------------------------------------------------------------

    def image_quality(self, quiet=False):
        
        res_score = iqa_score(self.tmp_img)

        if not quiet:
            QtWidgets.QMessageBox.information(None, "Image Viewer", "Image Quality (lower=better): %.3f" % res_score)

        return res_score

    # -----------------------------------------------------------------------------------------

    def hide_params(self):
        for widget in self.parameter_widgets:
            widget.hide()
        self.params_en = False
    
    # -----------------------------------------------------------------------------------------

    def show_params(self):
        for widget in self.parameter_widgets:
            widget.show()
        self.params_en = True

    # -----------------------------------------------------------------------------------------
    
    def toggle_params(self):
        self.hide_params() if self.params_en else self.show_params()

    # -----------------------------------------------------------------------------------------

    def quit(self):
        QtCore.QCoreApplication.instance().quit()

    # -----------------------------------------------------------------------------------------
    
    def loadImage(self):
        """ This function will load the user selected image
            and set it to label using the setPhoto function
        """
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.tif)")[0]
        if self.filename:
            bayer_img = cv2.imread(self.filename, cv2.IMREAD_UNCHANGED)
            try:
                self.image = cv2.cvtColor(bayer_img, cv2.COLOR_BAYER_BG2BGR)
            except Exception as e:
                QtWidgets.QMessageBox.information(None, "Image Viewer", "Cannot load %s --> %s" % (self.filename, str(e)))
                return
        
        self.setPhoto(self.image)
        self.scaleFactor = 1.0
        self.scrollArea.setVisible(True)
        self.printAct.setEnabled(True)
        self.fitToWindowAct.setEnabled(True)
        self.updateActions()
        if not self.fitToWindowAct.isChecked():
            self.label.adjustSize()
        self.show_params()
        self.pushButton.show()
        self.pushButton_3.show()
        self.pushButton_4.show()
        self.pushButton_5.show()
        
    # -----------------------------------------------------------------------------------------
    
    def setPhoto(self, image):
        """ This function will take image input and resize it 
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.tmp_img = image
        #image = imutils.resize(image, width=640)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1],frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))
    
    # -----------------------------------------------------------------------------------------
    
    def savePhoto(self):
        """ This function will save the image"""
        filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
        
        cv2.imwrite(filename, self.tmp_img)
        QtWidgets.QMessageBox.information(None, "Image Viewer", "Image saved as: %s" % filename)
    
    # -----------------------------------------------------------------------------------------
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Photo Editor"))
        self.pushButton.setText(_translate("MainWindow", "Save"))
        self.pushButton_2.setText(_translate("MainWindow", "Open"))
        self.pushButton_3.setText(_translate("MainWindow", "Launch"))
        self.pushButton_4.setText(_translate("MainWindow", "Toggle Params"))
        self.pushButton_5.setText(_translate("MainWindow", "Image Quality"))
        self.pushButton_6.setText(_translate("MainWindow", "Exit"))
        for param_name in self.defparams.keys():
            label_obj_name = '%s_label' % param_name
            label_obj = getattr(self, label_obj_name)
            label_obj.setText(_translate("MainWindow", param_name.capitalize())) 

    # -----------------------------------------------------------------------------------------

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.label.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.label.pixmap().rect())
            painter.drawPixmap(0, 0, self.label.pixmap())
    
    # -----------------------------------------------------------------------------------------

    def zoomIn(self):
        self.scaleImage(1.25)
    
    # -----------------------------------------------------------------------------------------

    def zoomOut(self):
        self.scaleImage(0.8)
    
    # -----------------------------------------------------------------------------------------

    def normalSize(self):
        self.label.adjustSize()
        self.scaleFactor = 1.0
    
    # -----------------------------------------------------------------------------------------

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()
    
    # -----------------------------------------------------------------------------------------

    def about(self):
        QMessageBox.about(self, "About Image Viewer",
                          "<p>The <b>Image Viewer</b> example shows how to combine "
                          "QLabel and QScrollArea to display an image. QLabel is "
                          "typically used for displaying text, but it can also display "
                          "an image. QScrollArea provides a scrolling view around "
                          "another widget. If the child widget exceeds the size of the "
                          "frame, QScrollArea automatically provides scroll bars.</p>"
                          "<p>The example demonstrates how QLabel's ability to scale "
                          "its contents (QLabel.scaledContents), and QScrollArea's "
                          "ability to automatically resize its contents "
                          "(QScrollArea.widgetResizable), can be used to implement "
                          "zooming and scaling features.</p>"
                          "<p>In addition the example shows how to use QPainter to "
                          "print an image.</p>")
    
    # -----------------------------------------------------------------------------------------

    def createActions(self, MainWindow):
        self.openAct = QtWidgets.QAction("&Open...", MainWindow, shortcut="Ctrl+O", triggered=self.loadImage)
        self.printAct = QtWidgets.QAction("&Print...", MainWindow, shortcut="Ctrl+P", enabled=False, triggered=self.print_)
        self.exitAct = QtWidgets.QAction("E&xit", MainWindow, shortcut="Ctrl+Q", triggered=self.quit)
        self.zoomInAct = QtWidgets.QAction("Zoom &In (25%)", MainWindow, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
        self.zoomOutAct = QtWidgets.QAction("Zoom &Out (25%)", MainWindow, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
        self.normalSizeAct = QtWidgets.QAction("&Normal Size", MainWindow, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)
        self.fitToWindowAct = QtWidgets.QAction("&Fit to Window", MainWindow, enabled=False, checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindow)
        self.aboutAct = QtWidgets.QAction("&About", MainWindow, triggered=self.about)
        self.aboutQtAct = QtWidgets.QAction("About &Qt", MainWindow, triggered=QtWidgets.qApp.aboutQt)
    
    # -----------------------------------------------------------------------------------------

    def createMenus(self, MainWindow):
        self.fileMenu = QtWidgets.QMenu("&File", MainWindow)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QtWidgets.QMenu("&View", MainWindow)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QtWidgets.QMenu("&Help", MainWindow)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        MainWindow.menuBar().addMenu(self.fileMenu)
        MainWindow.menuBar().addMenu(self.viewMenu)
        MainWindow.menuBar().addMenu(self.helpMenu)
    
    # -----------------------------------------------------------------------------------------

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())
    
    # -----------------------------------------------------------------------------------------

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.label.resize(self.scaleFactor * self.label.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 5.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.2)
    
    # -----------------------------------------------------------------------------------------

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep() / 2)))

# ===========================================================================================================


if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    MainWindow = QtWidgets.QMainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

