# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI_MMF7.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 970)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.EDA1 = QtWidgets.QPushButton(self.centralwidget)
        self.EDA1.setGeometry(QtCore.QRect(50, 820, 190, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.EDA1.sizePolicy().hasHeightForWidth())
        self.EDA1.setSizePolicy(sizePolicy)
        self.EDA1.setObjectName("EDA1")
        self.Modeling = QtWidgets.QPushButton(self.centralwidget)
        self.Modeling.setGeometry(QtCore.QRect(504, 870, 190, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Modeling.sizePolicy().hasHeightForWidth())
        self.Modeling.setSizePolicy(sizePolicy)
        self.Modeling.setObjectName("Modeling")
        self.Evaluation = QtWidgets.QPushButton(self.centralwidget)
        self.Evaluation.setGeometry(QtCore.QRect(731, 870, 190, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Evaluation.sizePolicy().hasHeightForWidth())
        self.Evaluation.setSizePolicy(sizePolicy)
        self.Evaluation.setObjectName("Evaluation")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 10, 1100, 800))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("Front.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.EDA3 = QtWidgets.QPushButton(self.centralwidget)
        self.EDA3.setGeometry(QtCore.QRect(504, 820, 190, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.EDA3.sizePolicy().hasHeightForWidth())
        self.EDA3.setSizePolicy(sizePolicy)
        self.EDA3.setObjectName("EDA3")
        ###
        self.EDA4 = QtWidgets.QPushButton(self.centralwidget)
        self.EDA4.setGeometry(QtCore.QRect(731, 820, 190, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.EDA4.sizePolicy().hasHeightForWidth())
        self.EDA4.setSizePolicy(sizePolicy)
        self.EDA4.setObjectName("EDA4")
        ###
        ###
        self.EDA5 = QtWidgets.QPushButton(self.centralwidget)
        self.EDA5.setGeometry(QtCore.QRect(958, 820, 190, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.EDA5.sizePolicy().hasHeightForWidth())
        self.EDA5.setSizePolicy(sizePolicy)
        self.EDA5.setObjectName("EDA5")
        ###
        ###
        self.FI = QtWidgets.QPushButton(self.centralwidget)
        self.FI.setGeometry(QtCore.QRect(50, 870, 190, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.FI.sizePolicy().hasHeightForWidth())
        self.FI.setSizePolicy(sizePolicy)
        self.FI.setObjectName("FI")
        ###
        ###
        self.FS = QtWidgets.QPushButton(self.centralwidget)
        self.FS.setGeometry(QtCore.QRect(277, 870, 190, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.FS.sizePolicy().hasHeightForWidth())
        self.FS.setSizePolicy(sizePolicy)
        self.FS.setObjectName("FS")
        ###

        self.EDA2 = QtWidgets.QPushButton(self.centralwidget)
        self.EDA2.setGeometry(QtCore.QRect(277, 820, 190, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.EDA2.sizePolicy().hasHeightForWidth())
        self.EDA2.setSizePolicy(sizePolicy)
        self.EDA2.setObjectName("EDA2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 719, 18))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.actionExit.triggered.connect(lambda: self.clicked())

        self.EDA1.clicked.connect(self.show_EDA1)
        self.EDA2.clicked.connect(self.show_EDA2)
        self.EDA3.clicked.connect(self.show_EDA3)
        self.EDA4.clicked.connect(self.show_EDA4)
        self.EDA5.clicked.connect(self.show_EDA5)
        self.FI.clicked.connect(self.show_FI)
        self.FS.clicked.connect(self.show_FS)
        self.Modeling.clicked.connect(self.show_Modeling)
        self.Evaluation.clicked.connect(self.show_Evaluation)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Predicition of the Breat Tumor Diagnosis"))
        self.EDA1.setText(_translate("MainWindow", "EDA - Class Labels"))
        self.Modeling.setText(_translate("MainWindow", "Modeling"))
        self.Evaluation.setText(_translate("MainWindow", "Evaluation"))
        self.EDA3.setText(_translate("MainWindow", "EDA - Features Part II"))
        self.EDA2.setText(_translate("MainWindow", "EDA - Features Part I"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.EDA4.setText(_translate("MainWindow", "EDA - Features Part III"))
        self.EDA5.setText(_translate("MainWindow", "EDA - Features Part IV"))
        self.FI.setText(_translate("MainWindow", "Feature Importance"))
        self.FS.setText(_translate("MainWindow", "Feature Selection"))

    def show_EDA1(self):
        self.label.setPixmap(QtGui.QPixmap("countplot_target.png"))

    def show_EDA2(self):
        self.label.setPixmap(QtGui.QPixmap("boxplots_worst.png"))

    def show_EDA3(self):
        self.label.setPixmap(QtGui.QPixmap("histograms_worst.png"))

    def show_EDA4(self):
        self.label.setPixmap(QtGui.QPixmap("polar_worst.png"))

    def show_EDA5(self):
        self.label.setPixmap(QtGui.QPixmap("corr_worst.png"))

    def show_FI(self):
        self.label.setPixmap(QtGui.QPixmap("feature_importance1.png"))

    def show_FS(self):
        self.label.setPixmap(QtGui.QPixmap("feature_selection.png"))

    def show_Modeling(self):
            self.label.setPixmap(QtGui.QPixmap("best_parameters.png"))

    def show_Evaluation(self):
            self.label.setPixmap(QtGui.QPixmap("best_results.png"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
