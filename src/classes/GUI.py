import sys
import datetime
import os.path
import re
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.Qt import QFont, Qt
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QLabel, QLineEdit, QCheckBox, QTableWidget,
                             QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout, QTableWidgetItem, QSpacerItem,
                             QSizePolicy, QFileDialog, QStackedWidget)
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import pyqtSignal
import random

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Prelearn')
        self.resize(1000, 500)

        self.buildPages()

    def buildPages(self):
        self.pages = Page()
        self.setCentralWidget(self.pages)

class Page(QStackedWidget):
    def __init__(self):
        super().__init__()

        self.startPage = StartPage()
        self.statisticPage = StatisticPage()
        self.resultPage = ResultPage()

        self.addWidget(self.startPage)
        self.addWidget(self.statisticPage)
        self.addWidget(self.resultPage)

        self.startPage.startRequest1.connect(lambda: self.setCurrentIndex(2))  # startPage -> resultPage
        self.startPage.startRequest2.connect(lambda: self.setCurrentIndex(1)) # startPage -> statisticPage
        self.statisticPage.statisticRequest1.connect(lambda: self.setCurrentIndex(0))  # statisticPage -> startPage
        self.resultPage.resultRequest1.connect(lambda: self.setCurrentIndex(0))  # resultPage -> startPage
        self.resultPage.resultRequest2.connect(lambda: self.setCurrentIndex(1))  # resultPage -> statisticPage

class StartPage(QWidget):
    startRequest1 = pyqtSignal()
    startRequest2 = pyqtSignal()

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

        self.createTable()

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
        self.startButton.clicked.connect(self.startRequest1)

        self.verticalSpacer = QSpacerItem(0, 500, QSizePolicy.Ignored, QSizePolicy.Ignored)

        buttonLayout = QVBoxLayout()
        buttonLayout.addItem(self.verticalSpacer)
        buttonLayout.addWidget(self.startButton)
        self.startButtonWidget.setLayout(buttonLayout)

    def createTable(self):
        if os.path.isfile('saved.txt'):
            file = open('saved.txt', 'r')
            fileLength = len(file.readlines())
            if fileLength > 0:
                self.rows = fileLength
            file.close()
        else:
            self.rows = 0
        self.rows = 4   # #TODO: da rimuovere, ora solo per provare funzionamento
        self.columns = 4
        self.startTable = QTableWidget(self.rows, self.columns)
        self.startTable.setHorizontalHeaderLabels(
            ['', 'Data', 'Performance', 'Parametri'])
        self.startTable.setColumnWidth(0, 30)
        self.startTable.setGeometry(300, 300, 250, 250)
        self.startTable.setDisabled(True)
        for row in range(self.rows):
            startRightCheckBoxItem = QTableWidgetItem(row)
            startRightCheckBoxItem.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            startRightCheckBoxItem.setCheckState(Qt.Unchecked)
            self.startTable.setItem(row, 0, startRightCheckBoxItem)
        if os.path.isfile('saved.txt'):
            file = open('saved.txt', 'r')
            for line in file:
                fields = line.split('\t')
                self.startTable.setItem(row, 0, QTableWidgetItem(fields[0]))
                self.startTable.setItem(row, 1, QTableWidgetItem(fields[1]))
                self.startTable.setItem(row, 2, QTableWidgetItem(fields[2]))

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
        options = QFileDialog().Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Scegliere il dataset", "", "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.startLeftDatasetLabel.setText('Dataset caricato: \n' + os.path.basename(fileName))

    def runModel(self):  # TODO: to run model with inserted or selected parameters
        pass

class StatisticPage(QWidget):
    statisticRequest1 = pyqtSignal()
    statisticRequest2 = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.createAccuracyGraphWidget()
        self.createPrecisionGraphWidget()
        self.createFScoreGraphWidget()
        self.createRecallGraphWidget()
        self.createResultReturnButton()

        statisticLayout = QGridLayout()
        statisticLayout.addWidget(self.accuracyWidget, 0, 0)
        statisticLayout.addWidget(self.precisionWidget, 0, 1)
        statisticLayout.addWidget(self.fscoreWidget, 1, 0)
        statisticLayout.addWidget(self.recallWidget, 1, 1)
        statisticLayout.addWidget(self.returnWidget, 2, 0)
        self.setLayout(statisticLayout)

    def createAccuracyGraphWidget(self):
        self.accuracyWidget = QWidget()

        self.accuracyLabel = Label('Accuracy:')
        self.accuracyGraph = Canvas()

        accuracyLayout = QVBoxLayout()
        accuracyLayout.addWidget(self.accuracyLabel)
        accuracyLayout.addWidget(self.accuracyGraph)
        self.accuracyWidget.setLayout(accuracyLayout)

    def createPrecisionGraphWidget(self):
        self.precisionWidget = QWidget()

        self.precisionLabel = Label('Precision:')
        self.precisionGraph = Canvas()

        precisionLayout = QVBoxLayout()
        precisionLayout.addWidget(self.precisionLabel)
        precisionLayout.addWidget(self.precisionGraph)
        self.precisionWidget.setLayout(precisionLayout)

    def createFScoreGraphWidget(self):
        self.fscoreWidget = QWidget()

        self.fscoreLabel = Label('FScore:')
        self.fscoreGraph = Canvas()

        fscoreLayout = QVBoxLayout()
        fscoreLayout.addWidget(self.fscoreLabel)
        fscoreLayout.addWidget(self.fscoreGraph)
        self.fscoreWidget.setLayout(fscoreLayout)

    def createRecallGraphWidget(self):
        self.recallWidget = QWidget()

        self.recallLabel = Label('Recall:')
        self.recallGraph = Canvas()

        recallLayout = QVBoxLayout()
        recallLayout.addWidget(self.recallLabel)
        recallLayout.addWidget(self.recallGraph)
        self.recallWidget.setLayout(recallLayout)

    def createResultReturnButton(self):
        self.returnWidget = QWidget()

        self.resultReturnButton = QPushButton()
        self.resultReturnButton.setText('Torna alla pagina iniziale')
        self.resultReturnButton.setFixedSize(200, 30)
        self.resultReturnButton.clicked.connect(self.statisticRequest1)

        returnLayout = QVBoxLayout()
        returnLayout.addWidget(self.resultReturnButton)
        self.returnWidget.setLayout(returnLayout)

class Canvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)

        self.plot()

    def plot(self):
        data = [random.random() for i in range(10)]
        ax = self.figure.add_subplot(111)
        ax.plot(data, 'r-')
        self.draw()

class ResultPage(QWidget):
    resultRequest1 = pyqtSignal()
    resultRequest2 = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.createResultLeftWidget()
        self.createResultRightWidget()
        self.createSaveWidget()
        self.createResultButtonWidget()

        resultPageLayout = QVBoxLayout()
        resultPageLayout.addWidget(self.resultLeftWidget)
        resultPageLayout.addWidget(self.saveWidget)
        resultPageLayout.addWidget(self.returnWidget)
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
        
        self.resultRightLabel1 = Label('Statistiche del dataset:')
        self.resultRightLabel2 = Label('Numero voci: ')
        #TODO: add list of statistics

    def createSaveWidget(self):
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
        #self.saveNoButton.clicked.connect() #TODO: function of no button

        saveLayout = QGridLayout()
        saveLayout.addWidget(self.saveLabel1, 0, 0, 0, 3)
        saveLayout.addWidget(self.saveYesButton, 1, 0)
        saveLayout.addWidget(self.saveNoButton, 1, 1)
        saveLayout.addItem(self.saveHSpacer, 1, 2)
        self.saveWidget.setLayout(saveLayout)

    def createResultButtonWidget(self):
        self.returnWidget = QWidget()

        self.resultReturnButton = QPushButton()
        self.resultReturnButton.setText('Torna alla pagina iniziale')
        self.resultReturnButton.setFixedSize(200, 30)
        self.returnVSpacer = QSpacerItem(0, 100, QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        self.resultStatisticButton = QPushButton()
        self.resultStatisticButton.setText('Statistiche delle configurazioni')
        self.resultStatisticButton.setFixedSize(200, 30)        

        self.resultReturnButton.clicked.connect(self.resultRequest1)
        self.resultStatisticButton.clicked.connect(self.resultRequest2)

        returnLayout = QVBoxLayout()
        returnLayout.addItem(self.returnVSpacer)
        returnLayout.addWidget(self.resultStatisticButton)
        returnLayout.addWidget(self.resultReturnButton)
        self.returnWidget.setLayout(returnLayout)

    def saveConfiguration(self):
        saveDate = datetime.datetime.now()
        # data conf, performance(label,valore), parametri(label,valore)
        file = open('saved.txt', 'a')
        file.write(str(saveDate.year) + '-' + str(saveDate.month) +
                   '-' + str(saveDate.day) + '\t')
        file.write('(' + self.resultLeftLabel2.text() + ',' + 'valore' + '),' +
                   '(' + self.resultLeftLabel3.text() + ',' + 'valore' + '),' +
                   '(' + self.resultLeftLabel4.text() + ',' + 'valore' + '),' +
                   '(' + self.resultLeftLabel5.text() + ',' + 'valore' + ')\t')
        file.write('(' + self.startLeftLabel2.text() + ',' + self.startLeftLineEdit1.text() + '),' +
                   '(' + self.startLeftLabel3.text() + ',' + self.startLeftLineEdit2.text() + '),' +
                   '(' + self.startLeftLabel4.text() + ',' + self.startLeftLineEdit3.text() + ')\n')
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

def main():
    app = QApplication(sys.argv)
    win = Window()
    # win.showMaximized() : to have screen window
    win.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
