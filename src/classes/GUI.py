import sys
import datetime
import os.path
import re
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.Qt import QFont, Qt
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QLabel, QLineEdit, QCheckBox, QTableWidget,
                             QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout, QTableWidgetItem, QSpacerItem,
                             QSizePolicy, QFileDialog, QStackedWidget, QHeaderView, QMenuBar, QAction, QDialog,
                             QTextBrowser, QTextEdit)
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import pyqtSignal
from Parser import Parser
from Engine import Engine
from Model import Model
from Settings import Settings

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Prelearn')
        self.resize(1000, 600)

        self.createMenu()
        self.buildPages()

    def buildPages(self):
        self.pages = Page()
        self.setCentralWidget(self.pages)
    
    def createMenu(self):
        guideAction = QAction('Guida', self)
        guideAction.triggered.connect(self.openGuideDialog)

        self.toolBar = self.addToolBar('Help')
        self.toolBar.addAction(guideAction)

    def openGuideDialog(self):
        self.guideDialog = Guide()
        self.guideDialog.show()


class Guide(QDialog):
    guidePage = Settings.guidePage

    def __init__(self):
        super().__init__()

        self.resize(400, 400)
        self.setWindowTitle('Guida')

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
        #self.resultPage.updateTable.connect(lambda: self.startPage.createTableWidget())

class StartPage(QWidget):
    counter = 0
    modelResult = {}
    startRequest1 = pyqtSignal()
    startRequest2 = pyqtSignal()
    updateResult = pyqtSignal()
    savedConfigurations = Settings.savedConfigurations

    def __init__(self):
        super().__init__()

        self.createStartLeftWidget()
        self.createStartRightWidget()
        self.createStartButtonWidget()

        startPageLayout = QGridLayout()
        startPageLayout.addWidget(self.startLeftWidget, 0, 0, 1, 0)
        startPageLayout.addWidget(self.startRightWidget, 0, 1, 3, 2)
        startPageLayout.addWidget(self.startButtonWidget, 1, 0, 2, 0)
        self.setLayout(startPageLayout)

    def createStartLeftWidget(self):
        self.startLeftWidget = QWidget()

        self.startLeftFileButton = QPushButton()
        self.startLeftFileButton.setText('Premi per caricare il dataset')
        self.startLeftFileButton.setCheckable(True)
        self.startLeftFileButton.setFixedSize(200, 30)
        self.startLeftFileButton.clicked.connect(self.loadDataset)

        self.startLeftDatasetLabel = Label('')

        self.startLeftLabel1 = Label('Inserire i parametri del modello:')
        self.startLeftLabel2 = Label('Numero di neuroni:')
        self.startLeftLabel3 = Label('Numero di layers della rete neurale:')
        self.startLeftLabel4 = Label('Numero di kfold splits:')
        self.startLeftLabel5 = Label('Numero di epoche del training per layer:')

        self.startLeftLineEdit1 = LineEdit()
        self.startLeftLineEdit2 = LineEdit()
        self.startLeftLineEdit3 = LineEdit()
        self.startLeftLineEdit4 = LineEdit()

        self.startLeftLabel6 = Label('Selezionare le features per il modello:')
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

        startLeftLayout = QVBoxLayout()
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
        self.startLeftWidget.setLayout(startLeftLayout)

    def createStartRightWidget(self):
        self.startRightWidget = QWidget()

        self.startRightLabel1 = Label('Caricare una configurazione precedente?')

        self.startRightCheckBox = QCheckBox()
        self.startRightCheckBox.setCheckable(True)
        self.startRightCheckBox.setChecked(False)
        self.startRightCheckBox.stateChanged.connect(self.enableTable)

        self.startRightButton = QPushButton()
        self.startRightButton.setText('Statistiche delle configurazioni')
        self.startRightButton.setFixedSize(200, 30)
        self.startRightButton.clicked.connect(self.startRequest2)

        self.createTableWidget()

        startRightLayout = QGridLayout()
        startRightLayout.addWidget(self.startRightLabel1, 0, 0)
        startRightLayout.addWidget(self.startRightCheckBox, 0, 1)
        startRightLayout.addWidget(self.startRightButton, 0, 2)
        startRightLayout.addWidget(self.startTable, 1, 0, 1, 4)
        self.startRightWidget.setLayout(startRightLayout)

    def createStartButtonWidget(self):
        self.startButtonWidget = QWidget()

        self.startButton = QPushButton(self)
        self.startButton.setText('Calcolo modello')
        self.startButton.setFixedSize(200, 30)
        self.startButton.clicked.connect(self.runModel)

        self.verticalSpacer = QSpacerItem(0, 500, QSizePolicy.Ignored, QSizePolicy.Ignored)

        buttonLayout = QVBoxLayout()
        buttonLayout.addItem(self.verticalSpacer)
        buttonLayout.addWidget(self.startButton)
        self.startButtonWidget.setLayout(buttonLayout)

    def createTableWidget(self):
        if os.path.exists(self.savedConfigurations):
            file = open(self.savedConfigurations, 'r')
            fileLength = len(file.readlines())
            if fileLength > 0:
                self.rows = fileLength
            file.close()
        else:
            self.rows = 0
        self.columns = 4
        self.startTable = QTableWidget(self.rows, self.columns)
        self.startTable.setHorizontalHeaderLabels(['', 'Data', 'Performance', 'Parametri'])
        self.startTable.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.startTable.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.startTable.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.startTable.setColumnWidth(0, 30)
        self.startTable.setGeometry(300, 300, 250, 250)
        self.startTable.setDisabled(True)
        if os.path.exists(self.savedConfigurations):
            file = open(self.savedConfigurations, 'r')
            fields = []
            for line in file:
                element = line.split('\t')
                fields.append(element)
            for row in range(0, len(fields)):
                startRightCheckBoxItem = QTableWidgetItem(row)
                startRightCheckBoxItem.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                startRightCheckBoxItem.setCheckState(Qt.Unchecked)
                self.startTable.setItem(row, 0, startRightCheckBoxItem)
                self.startTable.setItem(row, 1, QTableWidgetItem(fields[row][0]))
                self.startTable.item(row, 1).setFlags(Qt.ItemIsEditable)
                self.startTable.setItem(row, 2, QTableWidgetItem(fields[row][1]))
                self.startTable.item(row, 2).setFlags(Qt.ItemIsEditable)
                self.startTable.setItem(row, 3, QTableWidgetItem(fields[row][2]))
                self.startTable.item(row, 3).setFlags(Qt.ItemIsEditable)
            file.close()

        self.startTable.itemClicked.connect(self.selectConfiguration)

    def enableTable(self):
        if self.startRightCheckBox.isChecked():
            self.startTable.setDisabled(False)
        else:
            self.startTable.setDisabled(True)

    def selectConfiguration(self, item):
        if item.checkState() == Qt.Checked:
            values = re.findall('[0-9]+', self.startTable.item(item.row(), 3).text())
            self.startLeftLineEdit1.setText(values[0])
            self.startLeftLineEdit2.setText(values[1])
            self.startLeftLineEdit3.setText(values[2])
            self.startLeftLineEdit4.setText(values[3])

    def loadDataset(self):
        fileDialog = QFileDialog(None, Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Scegliere il dataset", "", "Text Files (*.txt);;CSV Files (*.csv)")
        if fileName:
            self.startLeftDatasetLabel.setText('Dataset caricato: \n' + os.path.basename(fileName))

    def runModel(self):
        self.counter += 1
        if self.counter == 1:
            # Parse files in Specified folder, optionally we can add input to modify Settings.resourcePath
            p = Parser()
            p.parse()
            p.parseTest()
            Settings.logger.info('Finished Parsing')

            # Calculate Baseline Performance
            '''
            base = Baseline()
            basePerformance = base.process()
            '''
            # Calculate Engine Performance            
            engine = Engine()
            if self.startLeftLineEdit1.text():
                Settings.neurons = float(self.startLeftLineEdit1.text())
            if self.startLeftLineEdit2.text():
                Settings.layers = float(self.startLeftLineEdit2.text())
            if self.startLeftLineEdit3.text():
                Settings.kfoldSplits = float(self.startLineEdit3.text())
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
            self.modelResult = engine.process() # might be cv results or testSet predictions, depending on Settings.generateOutput
            if Settings.useCache:
                p.cache()
            self.startButton.clicked.connect(self.updateResult)
            self.startButton.setText('Vai ai risultati')    
        
        self.startButton.clicked.connect(self.startRequest1)


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
                values = fields[1].split(')')
                accuracy[fields[0]] = float(values[0].split(',')[-1])
                precision[fields[0]] = float(values[1].split(',')[-1])
                fscore[fields[0]] = float(values[2].split(',')[-1])
                recall[fields[0]] = float(values[3].split(',')[-1])
            file.close()
        self.graphWidget = QWidget()

        self.accuracyLabel = Label('Accuracy:')
        self.accuracyGraph = Graph()
        self.accuracyGraph.plot(accuracy.keys(), accuracy.values())

        self.precisionLabel = Label('Precision:')
        self.precisionGraph = Graph(self, width=5, height=4)
        self.precisionGraph.plot(precision.keys(), precision.values())

        self.fscoreLabel = Label('FScore:')
        self.fscoreGraph = Graph(self, width=5, height=4)
        self.fscoreGraph.plot(fscore.keys(), fscore.values())

        self.recallLabel = Label('Recall:')
        self.recallGraph = Graph(self, width=5, height=4)
        self.recallGraph.plot(recall.keys(), recall.values())

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
        self.statisticReturnButton.setText('Torna alla pagina iniziale')
        self.statisticReturnButton.setFixedSize(200, 30)
        self.statisticReturnButton.clicked.connect(self.statisticRequest1)

        self.statisticResultButton = QPushButton()
        self.statisticResultButton.setText('Torna alla pagina dei risultati')
        self.statisticResultButton.setFixedSize(200, 30)
        self.statisticResultButton.clicked.connect(self.statisticRequest2)

        returnLayout = QHBoxLayout()
        returnLayout.addWidget(self.statisticReturnButton)
        returnLayout.addWidget(self.statisticResultButton)
        self.returnWidget.setLayout(returnLayout)


class Graph(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

    def plot(self, x, y):
        ax = self.figure.add_subplot(111)
        ax.bar(x, y, color='blue', width=0.2)
        ax.set_ylim(0, 1)
        self.draw()


class ResultPage(QWidget):
    resultRequest1 = pyqtSignal()
    resultRequest2 = pyqtSignal()
    updateTable = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent

        self.createResultLeftWidget()
        #self.createResultRightWidget()
        self.createResultShowWidget()
        self.createResultSaveWidget()
        self.createResultButtonWidget()

        resultPageLayout = QGridLayout()
        resultPageLayout.addWidget(self.resultLeftWidget, 0, 0)
        resultPageLayout.addWidget(self.resultShowWidget, 0, 1, 1, 1)
        resultPageLayout.addWidget(self.saveWidget, 1, 0)
        resultPageLayout.addWidget(self.returnWidget, 2, 0)
        #resultPageLayout.addWidget(self.resultRightWidget, 0, 1, 1, 1)
        self.setLayout(resultPageLayout)

    def createResultLeftWidget(self):
        self.resultLeftWidget = QWidget()
        
        self.resultLeftLabel1 = Label('Risultato del calcolo sul modello configurato:')
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
    '''
    def createResultRightWidget(self):
        self.resultRightWidget = QWidget()

        self.resultRightLabel1 = Label('Dataset:')
        self.resultRightLabel2 = Label('Numero voci: ')
        self.resultRightLabel3 = Label('Numero domini: ')
        self.resultRightLabel4 = Label('Numero parole: ')
        self.resultHSpacer = QSpacerItem(400, 0, QSizePolicy.Maximum, QSizePolicy.Maximum)
        #TODO: add list of statistics

        resultRightLayout = QVBoxLayout()
        resultRightLayout.addWidget(self.resultRightLabel1)
        resultRightLayout.addWidget(self.resultRightLabel2)
        resultRightLayout.addWidget(self.resultRightLabel3)
        resultRightLayout.addItem(self.resultHSpacer)
        self.resultRightWidget.setLayout(resultRightLayout)
    '''
    def createResultShowWidget(self):
        self.resultShowWidget = QWidget()

        self.resultShowButton1 = QPushButton()
        self.resultShowButton1.setText('Mostra risultati')
        self.resultShowButton1.setFixedSize(100, 30)

        self.resultShowButton2 = QPushButton()
        self.resultShowButton2.setText('Salva risultati(txt)')
        self.resultShowButton2.setFixedSize(100, 30)
        
        self.resultShowButton3 = QPushButton()
        self.resultShowButton3.setText('Salva risultati(csv)')
        self.resultShowButton3.setFixedSize(100, 30)

        self.resultShowHSpacer = QSpacerItem(500, 0, QSizePolicy.Maximum, QSizePolicy.Maximum)

        self.resultShowButton1.clicked.connect(self.showResult)
        self.resultShowButton2.clicked.connect(self.saveResultTxt)
        self.resultShowButton3.clicked.connect(self.saveResultCsv)

        resultShowLayout = QHBoxLayout()
        resultShowLayout.addWidget(self.resultShowButton1)
        resultShowLayout.addWidget(self.resultShowButton2)
        resultShowLayout.addWidget(self.resultShowButton3)
        resultShowLayout.addItem(self.resultShowHSpacer)
        self.resultShowWidget.setLayout(resultShowLayout)

    def createResultSaveWidget(self):
        self.saveWidget = QWidget()

        self.saveLabel1 = Label('Vuoi salvare la configurazione?')
        
        self.saveYesButton = QPushButton()
        self.saveYesButton.setText('Si')
        self.saveYesButton.setFixedSize(100, 30)

        self.saveNoButton = QPushButton()
        self.saveNoButton.setText('No')
        self.saveNoButton.setFixedSize(100, 30)

        self.saveLabel2 = Label('')

        self.saveHSpacer = QSpacerItem(500, 0, QSizePolicy.Maximum, QSizePolicy.Maximum)

        self.saveYesButton.clicked.connect(self.saveConfiguration)
        #self.saveYesButton.clicked.connect(self.updateTable)
        self.saveNoButton.clicked.connect(self.disableSaveConfiguration)

        saveLayout = QGridLayout()
        saveLayout.addWidget(self.saveLabel1, 0, 0, 1, 3)
        saveLayout.addWidget(self.saveYesButton, 1, 0)
        saveLayout.addWidget(self.saveNoButton, 1, 1)
        saveLayout.addWidget(self.saveLabel2, 2, 0)
        saveLayout.addItem(self.saveHSpacer, 3, 2)
        self.saveWidget.setLayout(saveLayout)

    def createResultButtonWidget(self):
        self.returnWidget = QWidget()

        self.statisticReturnButton = QPushButton()
        self.statisticReturnButton.setText('Torna alla pagina iniziale')
        self.statisticReturnButton.setFixedSize(200, 30)
        self.returnVSpacer = QSpacerItem(0, 100, QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        self.resultStatisticButton = QPushButton()
        self.resultStatisticButton.setText('Statistiche delle configurazioni')
        self.resultStatisticButton.setFixedSize(200, 30)        

        self.statisticReturnButton.clicked.connect(self.resultRequest1)
        self.resultStatisticButton.clicked.connect(self.resultRequest2)

        returnLayout = QVBoxLayout()
        returnLayout.addItem(self.returnVSpacer)
        returnLayout.addWidget(self.resultStatisticButton)
        returnLayout.addWidget(self.statisticReturnButton)
        self.returnWidget.setLayout(returnLayout)

    def updateResult(self):
        self.resultLeftLabel2.setText('Accuracy: ' + str(self.parent.modelResult['accuracy']))
        self.resultLeftLabel3.setText('Precision: ' + str(self.parent.modelResult['precision']))
        self.resultLeftLabel4.setText('Fscore: ' + str(self.parent.modelResult['f1']))
        self.resultLeftLabel5.setText('Recall: ' + str(self.parent.modelResult['recall']))

    def showResult(self):  # TODO: open new window with results (like guide)
        self.showDialog = QDialog()
        self.showDialog.resize(400, 400)
        self.showDialog.setWindowTitle('Risultati')

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
            self, "Salva i risultati", "", "Text Files (*.txt)")
        if not fileName[0] == "":
            file = open(fileName[0], 'w')
            for domain in self.parent.modelResult['result'].values():
                for element in domain:
                    file.write(element[0] + ',' + element[1] + ',' + str(element[2]) + '\n')
            file.close()
    
    def saveResultCsv(self):
        resultDialog = QFileDialog(None, Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        fileName = QFileDialog.getSaveFileName(
            self, "Salva i risultati", "", "CSV Files (*.csv)")
        if not fileName[0] == "":
            file = open(fileName[0], 'w')
            for domain in self.parent.modelResult['result'].values():
                for element in domain:
                    file.write(element[0] + ',' + element[1] + ',' + str(element[2]) + '\n')  # TODO: insert result of the model
            file.close()

    def saveConfiguration(self):
        saveDate = datetime.datetime.now()
        # data conf, performance(label,valore), parametri(label,valore)
        file = open(self.parent.savedConfigurations, 'a')
        file.write(str(saveDate.year) + '-' + str(saveDate.month) +
                   '-' + str(saveDate.day) + ' ' + str(saveDate.hour) +
                   ':' + str(saveDate.minute) + ':' + str(saveDate.second) + '\t')
        file.write('(' + 'Accuracy' + ',' + str(self.parent.modelResult['accuracy']) + '),' +
                   '(' + 'Precision' + ',' + str(self.parent.modelResult['precision']) + '),' +
                   '(' + 'Fscore' + ',' + str(self.parent.modelResult['f1']) + '),' +
                   '(' + 'Recall' + ',' + str(self.parent.modelResult['recall']) + ')\t')
        file.write('(' + self.parent.startLeftLabel2.text() + ',' + self.parent.startLeftLineEdit1.text() + '),' +
                   '(' + self.parent.startLeftLabel3.text() + ',' + self.parent.startLeftLineEdit2.text() + '),' +
                   '(' + self.parent.startLeftLabel4.text() + ',' + self.parent.startLeftLineEdit3.text() + '),' +
                   '(' + self.parent.startLeftLabel5.text() + ',' + self.parent.startLeftLineEdit4.text() + ')\n')
        file.close()
        self.parent.startTable.update()

    def disableSaveConfiguration(self):
        self.saveWidget.setDisabled(True)


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

def main():
    app = QApplication(sys.argv)
    win = Window()
    win.showMaximized()  # to have screen window
    win.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
