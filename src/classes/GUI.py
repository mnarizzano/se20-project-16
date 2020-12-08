import sys
import datetime
import os.path
import re
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.Qt import QFont, Qt, QSize
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QLabel, QLineEdit, QCheckBox, QTableWidget,
                             QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout, QTableWidgetItem, QSpacerItem,
                             QSizePolicy, QFileDialog, QStackedWidget, QHeaderView, QMenuBar, QAction, QDialog,
                             QTextBrowser, QTextEdit, QScrollBar, QScrollArea)
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import pyqtSignal, QThread
from Parser import Parser
import Engine
from Model import Model
from Settings import Settings

processingThread = None
guidePage = None
progressPage = None

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Prelearn')
        self.resize(1000, 600)

        self.buildPages()

    def closeEvent(self, QCloseEvent):
        # close pages that might be open
        global guidePage
        if guidePage != None:
            guidePage.reject()
        if self.pages.startPage.progressDialog != None:
            self.pages.startPage.progressDialog.reject()

    def buildPages(self):
        self.pages = Page()
        self.setCentralWidget(self.pages)


class Guide(QDialog):
    guidePage = Settings.guidePage

    def __init__(self):
        super().__init__()

        self.resize(400, 400)
        self.setWindowTitle('Guide')

        self.createGuideWidget()

        guideDialogLayout = QVBoxLayout()
        guideDialogLayout.addWidget(self.guideWidget)
        self.setLayout(guideDialogLayout)

    def createGuideWidget(self):
        self.guideWidget = QWidget()

        self.guideText = QTextBrowser()
        self.guideHtml = open(self.guidePage).read()
        self.guideText.setHtml(self.guideHtml)

        guideLayout = QVBoxLayout()
        guideLayout.addWidget(self.guideText)
        self.guideWidget.setLayout(guideLayout)

class Progress(QDialog):
    # progressPage = Settings.progressPage

    def __init__(self):
        super().__init__()

        self.resize(400, 400)
        self.setWindowTitle('Progress')

        self.createProgressWidget()

        progressDialogLayout = QVBoxLayout()
        progressDialogLayout.addWidget(self.progressWidget)
        self.setLayout(progressDialogLayout)

    def createProgressWidget(self):
        self.progressWidget = QWidget()

        self.progressText = QTextBrowser()
        # self.progressHtml = open(self.progressPage).read()
        self.progressText.setText("")

        progressLayout = QVBoxLayout()
        progressLayout.addWidget(self.progressText)
        self.progressWidget.setLayout(progressLayout)

class Page(QStackedWidget):
    def __init__(self):
        super().__init__()

        self.startPage = StartPage()
        self.statisticPage = StatisticPage(self.startPage)
        self.resultPage = ResultPage(self.startPage)

        self.addWidget(self.startPage)
        self.addWidget(self.statisticPage)
        self.addWidget(self.resultPage)

        self.startPage.startRequest1.connect(lambda: self.setCurrentIndex(2))  # startPage -> resultPage
        self.startPage.startRequest2.connect(lambda: self.setCurrentIndex(1)) # startPage -> statisticPage
        self.statisticPage.statisticRequest1.connect(lambda: self.setCurrentIndex(0))  # statisticPage -> startPage
        self.statisticPage.statisticRequest2.connect(lambda: self.setCurrentIndex(2)) # statisticPage -> resultPage
        self.resultPage.resultRequest1.connect(lambda: self.setCurrentIndex(0))  # resultPage -> startPage
        self.resultPage.resultRequest2.connect(lambda: self.setCurrentIndex(1))  # resultPage -> statisticPage

        self.startPage.updateResult.connect(lambda: self.resultPage.updateResult())

class StartPage(QWidget):
    modelResult = {}
    startRequest1 = pyqtSignal()
    startRequest2 = pyqtSignal()
    updateResult = pyqtSignal()
    savedConfigurations = Settings.savedConfigurations

    def __init__(self):
        super().__init__()

        self.createStartLeftWidget()
        self.createStartRightWidget()

        startPageLayout = QGridLayout()
        # self.startLeftWidget.setStyleSheet("background-color: rgb(255,0,0); margin:5px; border:1px solid rgb(0, 255, 0); ")
        startPageLayout.addWidget(self.startRightWidget, 0, 5, -1, 10)
        startPageLayout.addWidget(self.startLeftWidget, 0, 0, -1, 5)

        self.setLayout(startPageLayout)

    def createStartLeftWidget(self):
        self.startLeftWidget = QWidget()

        self.guideButton = QPushButton()
        self.guideButton.setText('Guide')
        self.guideButton.setCheckable(True)
        self.guideButton.setFixedSize(100, 30)
        self.guideButton.clicked.connect(self.openGuideDialog)

        self.startLeftFileButton = QPushButton()
        self.startLeftFileButton.setText('Load custom dataset')
        self.startLeftFileButton.setCheckable(True)
        self.startLeftFileButton.setFixedSize(200, 30)
        self.startLeftFileButton.clicked.connect(self.loadDataset)

        self.startLeftDatasetLabel = Label('')

        self.startLeftLabel1 = Label('Insert the parameters to configure the model:')
        self.startLeftLabel2 = Label('Neurons:')
        self.startLeftLabel3 = Label('Neural Network Layers:')
        self.startLeftLabel4 = Label('Kfold splits:')
        self.startLeftLabel5 = Label('Training epochs per layer:')

        self.startLeftLineEdit1 = LineEdit()
        self.startLeftLineEdit2 = LineEdit()
        self.startLeftLineEdit3 = LineEdit()
        self.startLeftLineEdit4 = LineEdit()

        self.startLeftLabel6 = Label('Select the features for the model:')
        self.startLeftCheckBox1 = QCheckBox('Reference Distance')
        self.startLeftCheckBox1.setCheckable(True)

        self.startLeftCheckBox2 = QCheckBox('Jaccard Similarity')
        self.startLeftCheckBox2.setCheckable(True)

        self.startLeftCheckBox3 = QCheckBox('LDA Concept')
        self.startLeftCheckBox3.setCheckable(True)

        self.startLeftCheckBox4 = QCheckBox('LDA Cross Entropy')
        self.startLeftCheckBox4.setCheckable(True)

        self.startLeftCheckBox5 = QCheckBox('LDA KLDivergence')
        self.startLeftCheckBox5.setCheckable(True)

        self.startLeftCheckBox6 = QCheckBox('Link')
        self.startLeftCheckBox6.setCheckable(True)

        self.startLeftCheckBox7 = QCheckBox('Nouns')
        self.startLeftCheckBox7.setCheckable(True)

        self.startLeftCheckBox8 = QCheckBox('Verbs')
        self.startLeftCheckBox8.setCheckable(True)

        self.startLeftCheckBox9 = QCheckBox('Adjectives')
        self.startLeftCheckBox9.setCheckable(True)

        self.startLeftCheckBox10 = QCheckBox('Crossdomain')
        self.startLeftCheckBox10.setCheckable(True)

        self.startLeftCheckBox11 = QCheckBox('Contains')
        self.startLeftCheckBox11.setCheckable(True)

        self.startLeftButton1 = QPushButton()
        self.startLeftButton1.setText('Run configuration')
        self.startLeftButton1.setFixedSize(200, 30)
        self.startLeftButton1.clicked.connect(self.runModel)

        self.startLeftButton2 = QPushButton()
        self.startLeftButton2.setText('Results')
        self.startLeftButton2.setFixedSize(200, 30)
        self.startLeftButton2.setDisabled(True)
        self.startLeftButton2.clicked.connect(self.updateResult)
        self.startLeftButton2.clicked.connect(self.startRequest1)

        self.verticalSpacer = QSpacerItem(0, 100, QSizePolicy.Ignored, QSizePolicy.Ignored)

        startLeftLayout = QVBoxLayout()
        startLeftLayout.addWidget(self.guideButton)
        #
        startLeftLayout.addWidget(self.startLeftFileButton)
        startLeftLayout.addWidget(self.startLeftDatasetLabel)
        startLeftLayout.addWidget(self.startLeftLabel1)
        startLeftLayout.addWidget(self.startLeftLabel2)
        startLeftLayout.addWidget(self.startLeftLineEdit1)
        startLeftLayout.addWidget(self.startLeftLabel3)
        startLeftLayout.addWidget(self.startLeftLineEdit2)
        startLeftLayout.addWidget(self.startLeftLabel4)
        startLeftLayout.addWidget(self.startLeftLineEdit3)
        startLeftLayout.addWidget(self.startLeftLabel5)
        startLeftLayout.addWidget(self.startLeftLineEdit4)
        startLeftLayout.addWidget(self.startLeftCheckBox1)
        startLeftLayout.addWidget(self.startLeftCheckBox2)
        startLeftLayout.addWidget(self.startLeftCheckBox3)
        startLeftLayout.addWidget(self.startLeftCheckBox4)
        startLeftLayout.addWidget(self.startLeftCheckBox5)
        startLeftLayout.addWidget(self.startLeftCheckBox6)
        startLeftLayout.addWidget(self.startLeftCheckBox7)
        startLeftLayout.addWidget(self.startLeftCheckBox8)
        startLeftLayout.addWidget(self.startLeftCheckBox9)
        startLeftLayout.addWidget(self.startLeftCheckBox10)
        startLeftLayout.addWidget(self.startLeftCheckBox11)
        #
        startLeftLayout.addItem(self.verticalSpacer)
        startLeftLayout.addWidget(self.startLeftButton1)
        startLeftLayout.addWidget(self.startLeftButton2)

        vBoxContainer = QWidget()
        vBoxContainer.setLayout(startLeftLayout)

        scrollArea = QScrollArea()
        scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollArea.setWidgetResizable(True)
        scrollArea.setWidget(vBoxContainer)
        self.startLeftWidget = scrollArea



    def createStartRightWidget(self):
        self.startRightWidget = QWidget()

        self.startRightLabel1 = Label('Load a previous configuration?')

        self.startRightCheckBox = QCheckBox()
        self.startRightCheckBox.setCheckable(True)
        self.startRightCheckBox.setChecked(False)
        self.startRightCheckBox.stateChanged.connect(self.enableTable)

        self.startRightButton = QPushButton()
        self.startRightButton.setText('Statistics')
        self.startRightButton.setFixedSize(200, 30)
        self.startRightButton.clicked.connect(self.startRequest2)

        self.createTableWidget()

        startRightLayout = QGridLayout()
        startRightLayout.addWidget(self.startRightLabel1, 0, 0)
        startRightLayout.addWidget(self.startRightCheckBox, 0, 1)
        startRightLayout.addWidget(self.startRightButton, 0, 2)
        startRightLayout.addWidget(self.startTable, 1, 0, 1, 4)
        self.startRightWidget.setLayout(startRightLayout)

    def createTableWidget(self):
        if os.path.exists(self.savedConfigurations):
            file = open(self.savedConfigurations, 'r')
            fileLength = len(file.readlines())
            if fileLength > 0:
                self.rows = fileLength
            file.close()
        else:
            self.rows = 0
        self.columns = 22
        self.startTable = QTableWidget(self.rows, self.columns)
        self.startTable.setHorizontalHeaderLabels(['','Date','Name','Accuracy','Precision','F-Score','Recall',
                                                    'Neurons','Layers','KfoldSplits','Epoch','RefD','JaccardSim',
                                                    'LDA Concept','LDA CrossEntropy','LDA KLD','Link','Nouns','Verbs',
                                                    'Adj','Crossdomain','Contains'])
        for i in range(0, 22):
            self.startTable.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.startTable.setGeometry(300, 300, 1000, 700)
        self.startTable.setDisabled(True)
        if os.path.exists(self.savedConfigurations):
            file = open(self.savedConfigurations, 'r')
            fileLines = file.readlines()
            for row in range(0, len(fileLines)):
                fields = fileLines[row].split('\t')
                performances = fields[2].split(')')
                parameters = fields[3].split(')')
                startRightCheckBoxItem = QTableWidgetItem(row)
                startRightCheckBoxItem.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                startRightCheckBoxItem.setCheckState(Qt.Unchecked)
                self.startTable.setItem(row, 0, startRightCheckBoxItem)
                self.startTable.setItem(row, 1, QTableWidgetItem(fields[0]))
                self.startTable.setItem(row, 2, QTableWidgetItem(fields[1]))
                self.startTable.setItem(row, 3, QTableWidgetItem(performances[0].split(',')[-1]))
                self.startTable.setItem(row, 4, QTableWidgetItem(performances[1].split(',')[-1]))                
                self.startTable.setItem(row, 5, QTableWidgetItem(performances[2].split(',')[-1]))
                self.startTable.setItem(row, 6, QTableWidgetItem(performances[3].split(',')[-1]))
                self.startTable.setItem(row, 7, QTableWidgetItem(parameters[0].split(',')[-1]))
                self.startTable.setItem(row, 8, QTableWidgetItem(parameters[1].split(',')[-1]))
                self.startTable.setItem(row, 9, QTableWidgetItem(parameters[2].split(',')[-1]))
                self.startTable.setItem(row, 10, QTableWidgetItem(parameters[3].split(',')[-1]))
                self.startTable.setItem(row, 11, QTableWidgetItem(parameters[4].split(',')[-1]))
                self.startTable.setItem(row, 12, QTableWidgetItem(parameters[5].split(',')[-1]))
                self.startTable.setItem(row, 13, QTableWidgetItem(parameters[6].split(',')[-1]))
                self.startTable.setItem(row, 14, QTableWidgetItem(parameters[7].split(',')[-1]))
                self.startTable.setItem(row, 15, QTableWidgetItem(parameters[8].split(',')[-1]))
                self.startTable.setItem(row, 16, QTableWidgetItem(parameters[9].split(',')[-1]))
                self.startTable.setItem(row, 17, QTableWidgetItem(parameters[10].split(',')[-1]))
                self.startTable.setItem(row, 18, QTableWidgetItem(parameters[11].split(',')[-1]))
                self.startTable.setItem(row, 19, QTableWidgetItem(parameters[12].split(',')[-1]))
                self.startTable.setItem(row, 20, QTableWidgetItem(parameters[13].split(',')[-1]))
                self.startTable.setItem(row, 21, QTableWidgetItem(parameters[14].split(',')[-1]))
            file.close()
        self.startTable.itemClicked.connect(self.selectConfiguration)

    def enableTable(self):
        if self.startRightCheckBox.isChecked():
            self.startTable.setDisabled(False)
        else:
            self.startTable.setDisabled(True)

    def selectConfiguration(self, item):
        if item.checkState() == Qt.Checked:
            for row in range(0, self.rows):
                if self.startTable.item(row, 0) != item:
                    self.startTable.item(row, 0).setCheckState(0)
                self.startLeftLineEdit1.setText(self.startTable.item(item.row(), 7).text())
                self.startLeftLineEdit2.setText(self.startTable.item(item.row(), 8).text())
                self.startLeftLineEdit3.setText(self.startTable.item(item.row(), 9).text())
                self.startLeftLineEdit4.setText(self.startTable.item(item.row(), 10).text())
                if self.startTable.item(item.row(), 11).text() == 'True':
                    self.startLeftCheckBox1.setCheckState(2)
                else:
                    self.startLeftCheckBox1.setCheckState(0)
                if self.startTable.item(item.row(), 12).text() == 'True':
                    self.startLeftCheckBox2.setCheckState(2)
                else:
                    self.startLeftCheckBox2.setCheckState(0)
                if self.startTable.item(item.row(), 13).text() == 'True':
                    self.startLeftCheckBox3.setCheckState(2)
                else:
                    self.startLeftCheckBox3.setCheckState(0)
                if self.startTable.item(item.row(), 14).text() == 'True':
                    self.startLeftCheckBox4.setCheckState(2)
                else:
                    self.startLeftCheckBox4.setCheckState(0)
                if self.startTable.item(item.row(), 15).text() == 'True':
                    self.startLeftCheckBox5.setCheckState(2)
                else:
                    self.startLeftCheckBox5.setCheckState(0)
                if self.startTable.item(item.row(), 16).text() == 'True':
                    self.startLeftCheckBox6.setCheckState(2)
                else: 
                    self.startLeftCheckBox6.setCheckState(0)
                if self.startTable.item(item.row(), 17).text() == 'True':
                    self.startLeftCheckBox7.setCheckState(2)
                else:
                    self.startLeftCheckBox7.setCheckState(0)
                if self.startTable.item(item.row(), 18).text() == 'True':
                    self.startLeftCheckBox8.setCheckState(2)
                else:
                    self.startLeftCheckBox8.setCheckState(0)
                if self.startTable.item(item.row(), 19).text() == 'True':
                    self.startLeftCheckBox9.setCheckState(2)
                else:
                    self.startLeftCheckBox9.setCheckState(0)
                if self.startTable.item(item.row(), 20).text() == 'True':
                    self.startLeftCheckBox10.setCheckState(2)
                else:
                    self.startLeftCheckBox10.setCheckState(0)
                if self.startTable.item(item.row(), 21).text() == 'True':
                    self.startLeftCheckBox11.setCheckState(2)
                else:
                    self.startLeftCheckBox11.setCheckState(0)

    def openGuideDialog(self):
        global guidePage
        guidePage = Guide()
        guidePage.show()

    def closeProgressDialog(self, text):
        self.progressDialog.hide()
        self.progressDialog.progressText.setText(text)

    def openProgressDialog(self):
        self.progressDialog = Progress()
        self.progressDialog.show()

    def setProgressText(self, text):
        self.progressDialog.setText(text)

    def loadDataset(self):
        fileDialog = QFileDialog(None, Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Select the dataset", "", "Text Files (*.txt);;CSV Files (*.csv)")
        if fileName:
            self.startLeftDatasetLabel.setText('Dataset caricato: \n' + os.path.basename(fileName))

    def runModel(self):
        # Calculate Engine performances
        engine = Engine.Engine()
        if self.startLeftLineEdit1.text():
            Settings.neurons = float(self.startLeftLineEdit1.text())
        if self.startLeftLineEdit2.text():
            Settings.layers = float(self.startLeftLineEdit2.text())
        if self.startLeftLineEdit3.text():
            Settings.kfoldSplits = float(self.startLeftLineEdit3.text())
        if self.startLeftLineEdit4.text():
            Settings.epoch = float(self.startLeftLineEdit4.text())
        if self.startLeftCheckBox1.isChecked():
            Settings.useRefD = True
        else:
            Settings.useRefD = False
        if self.startLeftCheckBox2.isChecked():
            Settings.useJaccard = True
        else:
            Settings.useJaccard = False
        if self.startLeftCheckBox3.isChecked():
            Settings.useConceptLDA = True
        else:
            Settings.useConceptLDA = False
        if self.startLeftCheckBox4.isChecked():
            Settings.useLDACrossEntropy = True
        else:
            Settings.useLDACrossEntropy = False
        if self.startLeftCheckBox5.isChecked():
            Settings.useLDA_KLDivergence = True
        else:
            Settings.useLDA_KLDivergence = False
        if self.startLeftCheckBox6.isChecked():
            Settings.useContainsLink = True
        else:
            Settings.useContainsLink = False
        if self.startLeftCheckBox7.isChecked():
            Settings.useNouns = True
        else:
            Settings.useNouns = False
        if self.startLeftCheckBox8.isChecked():
            Settings.useVerbs = True
        else:
            Settings.useVerbs = False
        if self.startLeftCheckBox9.isChecked():
            Settings.useAdjectives = True
        else:
            Settings.useAdjectives = False
        if self.startLeftCheckBox10.isChecked():
            Settings.CrossDomain = True
        else:
            Settings.CrossDomain = False
        if self.startLeftCheckBox11.isChecked():
            Settings.contains = True
        else:
            Settings.contains = False
        global processingThread
        self.openProgressDialog()
        processingThread = MainBackgroundThread(engine, self)
        processingThread.signalProgress.connect(self.progressDialog.progressText.append)
        processingThread.signalEnd.connect(self.closeProgressDialog)
        processingThread.start()
        #self.modelResult = engine.process(self.startLeftButton1) # might be cv results or testSet predictions, depending on Settings.generateOutput


class StatisticPage(QWidget):
    statisticRequest1 = pyqtSignal()
    statisticRequest2 = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent

        self.createGraphWidget()
        self.createStatisticButtonWidget()

        statisticLayout = QVBoxLayout()
        statisticLayout.addWidget(self.graphWidget)
        statisticLayout.addWidget(self.returnWidget)
        self.setLayout(statisticLayout)

    def createGraphWidget(self):
        accuracy = {}
        precision = {}
        fscore = {}
        recall = {}

        if os.path.exists(self.parent.savedConfigurations):
            file = open(self.parent.savedConfigurations, 'r')
            for line in file:
                fields = (line.split('\t'))
                values = fields[2].split(')')
                accuracy[fields[1]] = float(values[0].split(',')[-1])
                precision[fields[1]] = float(values[1].split(',')[-1])
                fscore[fields[1]] = float(values[2].split(',')[-1])
                recall[fields[1]] = float(values[3].split(',')[-1])
            file.close()
        self.graphWidget = QWidget()

        self.accuracyLabel = Label('Accuracy:')
        self.accuracyGraph = Graph()
        self.accuracyGraph.plot(accuracy.keys(), accuracy.values(), 'value of Accuracy')

        self.precisionLabel = Label('Precision:')
        self.precisionGraph = Graph()
        self.precisionGraph.plot(precision.keys(), precision.values(), 'value of Precision')

        self.fscoreLabel = Label('FScore:')
        self.fscoreGraph = Graph()
        self.fscoreGraph.plot(fscore.keys(), fscore.values(), 'value of Fscore')

        self.recallLabel = Label('Recall:')
        self.recallGraph = Graph()
        self.recallGraph.plot(recall.keys(), recall.values(), 'value of Recall')

        graphWidgetLayout = QGridLayout()
        graphWidgetLayout.addWidget(self.accuracyLabel, 0, 0)
        graphWidgetLayout.addWidget(self.accuracyGraph, 1, 0)
        graphWidgetLayout.addWidget(self.precisionLabel, 0, 1)
        graphWidgetLayout.addWidget(self.precisionGraph, 1, 1)
        graphWidgetLayout.addWidget(self.fscoreLabel, 2, 0)
        graphWidgetLayout.addWidget(self.fscoreGraph, 3, 0)
        graphWidgetLayout.addWidget(self.recallLabel, 2, 1)
        graphWidgetLayout.addWidget(self.recallGraph, 3, 1)
        self.graphWidget.setLayout(graphWidgetLayout)
        
    def createStatisticButtonWidget(self):
        self.returnWidget = QWidget()

        self.statisticReturnButton = QPushButton()
        self.statisticReturnButton.setText('Homepage')
        self.statisticReturnButton.setFixedSize(200, 30)
        self.statisticReturnButton.clicked.connect(self.statisticRequest1)

        returnLayout = QHBoxLayout()
        returnLayout.addWidget(self.statisticReturnButton)
        self.returnWidget.setLayout(returnLayout)


class Graph(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=75):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

    def plot(self, x, y, ylabel):
        ax = self.figure.add_subplot(111)
        ax.bar(x, y, color='blue', width=0.2)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Configurations title')
        ax.set_ylabel(ylabel)
        self.draw()


class ResultPage(QWidget):
    resultRequest1 = pyqtSignal()
    resultRequest2 = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent

        self.createResultLeftWidget()
        self.createResultShowWidget()
        self.createResultSaveWidget()
        self.createResultButtonWidget()

        resultPageLayout = QGridLayout()
        resultPageLayout.addWidget(self.resultLeftWidget, 0, 0)
        resultPageLayout.addWidget(self.resultShowWidget, 0, 1, 2, 2)
        resultPageLayout.addWidget(self.saveWidget, 1, 0)
        resultPageLayout.addWidget(self.returnWidget, 2, 0)
        self.setLayout(resultPageLayout)

    def createResultLeftWidget(self):
        self.resultLeftWidget = QWidget()
        
        self.resultLeftLabel1 = Label('Results of the loaded configuration:')
        self.resultLeftLabel2 = Label('Accuracy: ')
        self.resultLeftLabel3 = Label('Precision: ')
        self.resultLeftLabel4 = Label('Fscore: ')
        self.resultLeftLabel5 = Label('Recall: ')

        resultLeftLayout = QVBoxLayout()
        resultLeftLayout.addWidget(self.resultLeftLabel1)
        resultLeftLayout.addWidget(self.resultLeftLabel2)
        resultLeftLayout.addWidget(self.resultLeftLabel3)
        resultLeftLayout.addWidget(self.resultLeftLabel4)
        resultLeftLayout.addWidget(self.resultLeftLabel5)
        self.resultLeftWidget.setLayout(resultLeftLayout)

    def createResultShowWidget(self):
        self.resultShowWidget = QWidget()

        self.resultShowButton1 = QPushButton()
        self.resultShowButton1.setText('Show the results')
        self.resultShowButton1.setFixedSize(200, 30)

        self.resultShowButton2 = QPushButton()
        self.resultShowButton2.setText('Save the results (txt)')
        self.resultShowButton2.setFixedSize(200, 30)
        
        self.resultShowButton3 = QPushButton()
        self.resultShowButton3.setText('Save the results (csv)')
        self.resultShowButton3.setFixedSize(200, 30)

        self.resultShowHSpacer = QSpacerItem(500, 0, QSizePolicy.Maximum, QSizePolicy.Maximum)

        self.resultShowButton1.clicked.connect(self.showResult)
        self.resultShowButton2.clicked.connect(self.saveResultTxt)
        self.resultShowButton3.clicked.connect(self.saveResultCsv)

        resultShowLayout = QGridLayout()
        resultShowLayout.addWidget(self.resultShowButton1, 0, 0, 2, 1)
        resultShowLayout.addWidget(self.resultShowButton2, 1, 0, 2, 1)
        resultShowLayout.addWidget(self.resultShowButton3, 2, 0, 2, 1)
        resultShowLayout.addItem(self.resultShowHSpacer, 1, 0, 3, 1)
        self.resultShowWidget.setLayout(resultShowLayout)

    def createResultSaveWidget(self):
        self.saveWidget = QWidget()

        self.saveLabel1 = Label('Do you want to save the configuration?')
        
        self.saveLabel2 = Label('Name:')

        self.saveLineEdit = QLineEdit()    
        self.saveLineEdit.setFixedSize(100, 20)

        self.saveYesButton = QPushButton()
        self.saveYesButton.setText('Si')
        self.saveYesButton.setFixedSize(100, 30)

        self.saveNoButton = QPushButton()
        self.saveNoButton.setText('No')
        self.saveNoButton.setFixedSize(100, 30)

        self.saveLabel3 = Label('')

        self.saveHSpacer = QSpacerItem(500, 0, QSizePolicy.Maximum, QSizePolicy.Maximum)

        self.saveYesButton.clicked.connect(self.saveConfiguration)
        self.saveYesButton.clicked.connect(self.updateTable)
        self.saveNoButton.clicked.connect(self.disableSaveConfiguration)

        saveLayout = QGridLayout()
        saveLayout.addWidget(self.saveLabel1, 0, 0, 1, 3)
        saveLayout.addWidget(self.saveLabel2, 1, 0)
        saveLayout.addWidget(self.saveLineEdit, 1, 1)
        saveLayout.addWidget(self.saveYesButton, 2, 0)
        saveLayout.addWidget(self.saveNoButton, 2, 1)
        saveLayout.addWidget(self.saveLabel3, 3, 0, 1, 2)
        saveLayout.addItem(self.saveHSpacer, 3, 2)
        self.saveWidget.setLayout(saveLayout)

    def createResultButtonWidget(self):
        self.returnWidget = QWidget()

        self.statisticReturnButton = QPushButton()
        self.statisticReturnButton.setText('Homepage')
        self.statisticReturnButton.setFixedSize(200, 30)
        self.returnVSpacer = QSpacerItem(0, 100, QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        self.resultStatisticButton = QPushButton()
        self.resultStatisticButton.setText('Statistics')
        self.resultStatisticButton.setFixedSize(200, 30)        

        self.statisticReturnButton.clicked.connect(self.resultRequest1)
        self.resultStatisticButton.clicked.connect(self.resultRequest2)

        returnLayout = QVBoxLayout()
        returnLayout.addItem(self.returnVSpacer)
        returnLayout.addWidget(self.resultStatisticButton)
        returnLayout.addWidget(self.statisticReturnButton)
        self.returnWidget.setLayout(returnLayout)

    def updateResult(self):
        self.resultLeftLabel2.setText('Accuracy: ' + str(round(self.parent.modelResult['accuracy'],3)))
        self.resultLeftLabel3.setText('Precision: ' + str(round(self.parent.modelResult['precision'],3)))
        self.resultLeftLabel4.setText('Fscore: ' + str(round(self.parent.modelResult['f1'],3)))
        self.resultLeftLabel5.setText('Recall: ' + str(round(self.parent.modelResult['recall'],3)))

    def showResult(self):
        self.showDialog = QDialog()
        self.showDialog.resize(400, 400)
        self.showDialog.setWindowTitle('Results')

        self.showWidget = QWidget()
        self.showText = QTextEdit()

        for domain in self.parent.modelResult['result'].values():
            for element in domain:
                self.showText.append(element[0] + ',' + element[1] + ',' + str(element[2]) + '\n')

        showLayout = QVBoxLayout()
        showLayout.addWidget(self.showText)
        self.showWidget.setLayout(showLayout)

        showDialogLayout = QVBoxLayout()
        showDialogLayout.addWidget(self.showWidget)
        self.showDialog.setLayout(showDialogLayout)

        self.showDialog.show()

    def saveResultTxt(self):
        resultDialog = QFileDialog(None, Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        fileName = QFileDialog.getSaveFileName(
            self, "Save the results", "", "Text Files (*.txt)")
        if not fileName[0] == "":
            file = open(fileName[0], 'w')
            for domain in self.parent.modelResult['result'].values():
                for element in domain:
                    file.write(element[0] + ',' + element[1] + ',' + str(element[2]) + '\n')
            file.close()
    
    def saveResultCsv(self):
        resultDialog = QFileDialog(None, Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        fileName = QFileDialog.getSaveFileName(
            self, "Save the results", "", "CSV Files (*.csv)")
        if not fileName[0] == "":
            file = open(fileName[0], 'w')
            for domain in self.parent.modelResult['result'].values():
                for element in domain:
                    file.write(element[0] + ',' + element[1] + ',' + str(element[2]) + '\n')
            file.close()

    def saveConfiguration(self):
        saveDate = datetime.datetime.now()
        file = open(self.parent.savedConfigurations, 'a')
        file.write(str(saveDate.year) + '-' + str(saveDate.month) +
                   '-' + str(saveDate.day) + ' ' + str(saveDate.hour) +
                   ':' + str(saveDate.minute) + ':' + str(saveDate.second) + '\t')
        if not self.saveLineEdit.text() == "":
            file.write(self.saveLineEdit.text() + '\t')
        else:
            file.write('default ' + str(saveDate.year) + '-' + str(saveDate.month) +
                       '-' + str(saveDate.day) + ' ' + str(saveDate.hour) +
                       ':' + str(saveDate.minute) + ':' + str(saveDate.second) + '\t')
        file.write('(' + 'Accuracy' + ',' + str(round(self.parent.modelResult['accuracy'],3)) + '),' +
                   '(' + 'Precision' + ',' + str(round(self.parent.modelResult['precision'],3)) + '),' +
                   '(' + 'Fscore' + ',' + str(round(self.parent.modelResult['f1'],3)) + '),' +
                   '(' + 'Recall' + ',' + str(round(self.parent.modelResult['recall'],3)) + ')\t')
        file.write('(' + self.parent.startLeftLabel2.text() + ',' + self.parent.startLeftLineEdit1.text() + '),' +
                   '(' + self.parent.startLeftLabel3.text() + ',' + self.parent.startLeftLineEdit2.text() + '),' +
                   '(' + self.parent.startLeftLabel4.text() + ',' + self.parent.startLeftLineEdit3.text() + '),' +
                   '(' + self.parent.startLeftLabel5.text() + ',' + self.parent.startLeftLineEdit4.text() + '),' +
                   '(' + self.parent.startLeftCheckBox1.text() + ',')
        if self.parent.startLeftCheckBox1.isChecked():
            file.write('True')
        else:
            file.write('False')
        file.write('),(' + self.parent.startLeftCheckBox2.text() + ',')
        if self.parent.startLeftCheckBox2.isChecked():
            file.write('True')
        else:
            file.write('False')
        file.write('),(' + self.parent.startLeftCheckBox3.text() + ',')
        if self.parent.startLeftCheckBox3.isChecked():
            file.write('True')
        else:
            file.write('False')
        file.write('),(' + self.parent.startLeftCheckBox4.text() + ',')
        if self.parent.startLeftCheckBox4.isChecked():
            file.write('True')
        else:
            file.write('False')
        file.write('),(' + self.parent.startLeftCheckBox5.text() + ',')
        if self.parent.startLeftCheckBox5.isChecked():
            file.write('True')
        else:
            file.write('False')
        file.write('),(' + self.parent.startLeftCheckBox6.text() + ',')
        if self.parent.startLeftCheckBox6.isChecked():
            file.write('True')
        else:
            file.write('False')
        file.write('),(' + self.parent.startLeftCheckBox7.text() + ',')
        if self.parent.startLeftCheckBox7.isChecked():
            file.write('True')
        else:
            file.write('False')
        file.write('),(' + self.parent.startLeftCheckBox8.text() + ',')
        if self.parent.startLeftCheckBox8.isChecked():
            file.write('True')
        else:
            file.write('False')
        file.write('),(' + self.parent.startLeftCheckBox9.text() + ',')
        if self.parent.startLeftCheckBox9.isChecked():
            file.write('True')
        else:
            file.write('False')
        file.write('),(' + self.parent.startLeftCheckBox10.text() + ',')
        if self.parent.startLeftCheckBox10.isChecked():
            file.write('True')
        else:
            file.write('False')
        file.write('),(' + self.parent.startLeftCheckBox11.text() + ',')
        if self.parent.startLeftCheckBox11.isChecked():
            file.write('True')
        else:
            file.write('False')
        file.write(')\n')
        file.close()
        self.saveLabel3.setText('Saved configuration')

    def disableSaveConfiguration(self):
        self.saveWidget.setDisabled(True)

    def updateTable(self):
        self.parent.startTable.setRowCount(self.parent.startTable.rowCount()+1)
        file = open(self.parent.savedConfigurations, 'r')
        newCheckBoxItem = QTableWidgetItem(self.parent.startTable.rowCount())
        newCheckBoxItem.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        newCheckBoxItem.setCheckState(Qt.Unchecked)
        newConfiguration = file.readlines()[-1]
        newFields = newConfiguration.split('\t')
        newPerformances = newFields[2].split(')')
        newParameters = newFields[3].split(')')
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 0, newCheckBoxItem)
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 1, QTableWidgetItem(newFields[0]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 2, QTableWidgetItem(newFields[1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 3, QTableWidgetItem(newPerformances[0].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 4, QTableWidgetItem(newPerformances[1].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 5, QTableWidgetItem(newPerformances[2].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 6, QTableWidgetItem(newPerformances[3].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 7, QTableWidgetItem(newParameters[0].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 8, QTableWidgetItem(newParameters[1].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 9, QTableWidgetItem(newParameters[2].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 10, QTableWidgetItem(newParameters[3].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 11, QTableWidgetItem(newParameters[4].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 12, QTableWidgetItem(newParameters[5].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 13, QTableWidgetItem(newParameters[6].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 14, QTableWidgetItem(newParameters[7].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 15, QTableWidgetItem(newParameters[8].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 16, QTableWidgetItem(newParameters[9].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 17, QTableWidgetItem(newParameters[10].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 18, QTableWidgetItem(newParameters[11].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 19, QTableWidgetItem(newParameters[12].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 20, QTableWidgetItem(newParameters[13].split(',')[-1]))
        self.parent.startTable.setItem(self.parent.startTable.rowCount()-1, 21, QTableWidgetItem(newParameters[14].split(',')[-1]))
        file.close()


class Label(QLabel):
    def __init__(self, text):
        super().__init__()

        self.setText(text)
        self.setFont(QFont('Arial', 10))
        self.adjustSize()

class LineEdit(QLineEdit):
    def __init__(self):
        super().__init__()
        # QDoubleValidator(bottom,top,decimal)
        self.setValidator(QDoubleValidator(0.00, 1000.00, 2))
        self.setAlignment(Qt.AlignRight)
        self.setMaxLength(8)
        self.setFixedSize(100, 20)

class MainBackgroundThread(QThread):
    signalProgress = pyqtSignal(str)
    signalEnd = pyqtSignal(str)

    def __init__(self, engine, startPage):
        QThread.__init__(self)
        self.engine = engine
        self.startPage = startPage

    def run(self):
        previousTableState = self.startPage.startTable.isEnabled()
        self.startPage.startTable.setDisabled(True)
        self.startPage.startLeftWidget.setDisabled(True)
        # Parse files in Specified folder, optionally we can add input to modify Settings.resourcePath
        self.signalProgress.emit("Parsing dataset...")
        self.parser = Parser()
        self.parser.parse()
        self.parser.parseTest()
        Settings.logger.info('Finished Parsing')
        self.startPage.modelResult = self.engine.process(self.signalProgress)
        if Settings.useCache:
            self.signalProgress.emit("Caching dataset...")
            self.parser.cache()
        self.startPage.startTable.setDisabled(not previousTableState)
        self.startPage.startLeftButton2.setDisabled(False)
        self.startPage.startLeftWidget.setDisabled(False)
        self.signalEnd.emit("Finished")

def main():
    app = QApplication(sys.argv)
    win = Window()
    win.showMaximized()  # to have screen window
    win.show()
    def runApp():
        exitCode = app.exec_()
        global processingThread
        if processingThread != None:
            processingThread.exit()
        return exitCode
    sys.exit(runApp())


if __name__ == '__main__':
    main()
