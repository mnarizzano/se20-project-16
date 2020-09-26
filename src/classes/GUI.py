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
                             QTextBrowser)
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
        self.resize(1000, 500)

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

class StartPage(QWidget):
    startRequest1 = pyqtSignal()
    startRequest2 = pyqtSignal()
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

        self.startLeftLabel1 = Label('Inserire una nuova configurazione:')
        self.startLeftLabel2 = Label('Parametro 1')
        self.startLeftLabel3 = Label('Parametro 2')
        self.startLeftLabel4 = Label('Parametro 3')

        self.startLeftLineEdit1 = LineEdit()
        self.startLeftLineEdit2 = LineEdit()
        self.startLeftLineEdit3 = LineEdit()

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
                self.startTable.setItem(row, 2, QTableWidgetItem(fields[row][1]))
                self.startTable.setItem(row, 3, QTableWidgetItem(fields[row][2]))
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

    def loadDataset(self):
        fileDialog = QFileDialog(None, Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Scegliere il dataset", "", "Text Files (*.txt);;CSV Files (*.csv)")
        if fileName:
            self.startLeftDatasetLabel.setText('Dataset caricato: \n' + os.path.basename(fileName))

    def runModel(self):
        # Parse files in Specified folder, optionally we can add input to modify Settings.resourcePath
        p = Parser()
        p.parse()
        Settings.logger.info('Finished Parsing')
        # Calculate Baseline Performance
        
        base = Baseline()
        basePerformance = base.process()
        
        # Calculate Engine Performance
        engine = Engine()
        engine.process()
        # plot statistics
        engine.plot()

        if Settings.useCache:
            p.cache()
            
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
                values = re.findall('[0-9]+', fields[1])
                accuracy[fields[0]] = int(values[0])
                precision[fields[0]] = int(values[1])
                fscore[fields[0]] = int(values[2])
                recall[fields[0]] = int(values[3])
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
        ax.set_ylim(0, 100)
        self.draw()


class ResultPage(QWidget):
    resultRequest1 = pyqtSignal()
    resultRequest2 = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent

        self.createResultLeftWidget()
        self.createResultRightWidget()
        self.createResultSaveWidget()
        self.createResultButtonWidget()

        resultPageLayout = QGridLayout()
        resultPageLayout.addWidget(self.resultLeftWidget, 0, 0)
        resultPageLayout.addWidget(self.saveWidget, 1, 0)
        resultPageLayout.addWidget(self.returnWidget, 2, 0)
        resultPageLayout.addWidget(self.resultRightWidget, 0, 1, 1, 1)
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

    def createResultSaveWidget(self):
        self.saveWidget = QWidget()

        self.saveLabel1 = Label('Vuoi salvare la configurazione?')
        
        self.saveYesButton = QPushButton()
        self.saveYesButton.setText('Si')
        self.saveYesButton.setFixedSize(100, 30)

        self.saveNoButton = QPushButton()
        self.saveNoButton.setText('No')
        self.saveNoButton.setFixedSize(100, 30)
        self.saveHSpacer = QSpacerItem(500, 0, QSizePolicy.Maximum, QSizePolicy.Maximum)

        self.saveYesButton.clicked.connect(self.saveConfiguration)
        self.saveNoButton.clicked.connect(self.disableSaveConfiguration)

        saveLayout = QGridLayout()
        saveLayout.addWidget(self.saveLabel1, 0, 0, 0, 3)
        saveLayout.addWidget(self.saveYesButton, 1, 0)
        saveLayout.addWidget(self.saveNoButton, 1, 1)
        saveLayout.addItem(self.saveHSpacer, 1, 2)
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

    def showResult(self):  # TODO: open new window with results (like guide)
        self.showDialog = QDialog()
        self.showDialog.resize(400, 400)
        self.showDialog.setWindowTitle('Risultati')

        self.showDialog.show()

    def saveResult(self):
        resultDialog = QFileDialog(
            None, Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        fileName = QFileDialog.getSaveFileName(
            self, "Salva i risultati", "", "Text Files (*.txt)")
        file = open(fileName[0], 'w')
        file.write()  # TODO: insert result of the model
        file.close()

    def saveConfiguration(self):
        saveDate = datetime.datetime.now()
        # data conf, performance(label,valore), parametri(label,valore)
        file = open(self.parent.savedConfigurations, 'a')
        file.write(str(saveDate.year) + '-' + str(saveDate.month) +
                   '-' + str(saveDate.day) + ' ' + str(saveDate.hour) +
                   ':' + str(saveDate.minute) + ':' + str(saveDate.second) + '\t')
        file.write('(' + 'Accuracy' + ',' + 'valore' + '),' +
                   '(' + 'Precision' + ',' + 'valore' + '),' +
                   '(' + 'Fscore' + ',' + 'valore' + '),' +
                   '(' + 'Recall' + ',' + 'valore' + ')\t') #TODO: substitute valore con output of nn
        file.write('(' + self.parent.startLeftLabel2.text() + ',' + self.parent.startLeftLineEdit1.text() + '),' +
                   '(' + self.parent.startLeftLabel3.text() + ',' + self.parent.startLeftLineEdit2.text() + '),' +
                   '(' + self.parent.startLeftLabel4.text() + ',' + self.parent.startLeftLineEdit3.text() + ')\n')
        file.close()  

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
    # win.showMaximized() : to have screen window
    win.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
