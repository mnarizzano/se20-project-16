import sys
import datetime
import os.path
import re
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.Qt import QFont, Qt
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QLabel, QLineEdit, QCheckBox, QTableWidget,
                             QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout, QTableWidgetItem, QSpacerItem,
                             QSizePolicy, QFileDialog)
from PyQt5.QtGui import QDoubleValidator
import random

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Prelearn')
        self.resize(1000, 500)
        self.startPage = StartPage()
        self.statisticPage = StatisticPage()
        self.resultPage = ResultPage()
        self.setCentralWidget(self.startPage)

class StartPage(QWidget):
    def __init__(self):
        super().__init__()

        self.createLeftWidget()
        self.createRightWidget()
        self.createButton()

        startPageLayout = QGridLayout()
        startPageLayout.addWidget(self.leftWidget, 0, 0, 1, 0)
        startPageLayout.addWidget(self.rightWidget, 0, 1, 3, 2)
        startPageLayout.addWidget(self.buttonWidget, 1, 0, 2, 0)
        self.setLayout(startPageLayout)

    def createLeftWidget(self):
        self.leftWidget = QWidget()

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

        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.startLeftFileButton)
        leftLayout.addWidget(self.startLeftDatasetLabel)
        leftLayout.addWidget(self.startLeftLabel1)
        leftLayout.addWidget(self.startLeftLabel2)
        leftLayout.addWidget(self.startLeftLineEdit1)
        leftLayout.addWidget(self.startLeftLabel3)
        leftLayout.addWidget(self.startLeftLineEdit2)
        leftLayout.addWidget(self.startLeftLabel4)
        leftLayout.addWidget(self.startLeftLineEdit3)
        self.leftWidget.setLayout(leftLayout)

    def createRightWidget(self):
        self.rightWidget = QWidget()

        self.startRightLabel1 = Label(
            'Caricare una configurazione precedente?')

        self.startRightCheckBox = QCheckBox()
        self.startRightCheckBox.setCheckable(True)
        self.startRightCheckBox.setChecked(False)
        self.startRightCheckBox.stateChanged.connect(self.enableTable)

        self.createTable()

        rightLayout = QGridLayout()
        rightLayout.addWidget(self.startRightLabel1, 0, 0)
        rightLayout.addWidget(self.startRightCheckBox, 0, 1)
        rightLayout.addWidget(self.startTable, 1, 0, 1, 3)
        self.rightWidget.setLayout(rightLayout)

    def createButton(self):
        self.buttonWidget = QWidget()

        self.startPushButton = QPushButton(self)
        self.startPushButton.setText('Calcolo modello')
        self.startPushButton.setCheckable(True)
        self.startPushButton.setFixedSize(200, 30)
        self.startPushButton.clicked.connect(self.runModel)

        self.verticalSpacer = QSpacerItem(
            0, 500, QSizePolicy.Ignored, QSizePolicy.Ignored)

        buttonLayout = QVBoxLayout()
        buttonLayout.addItem(self.verticalSpacer)
        buttonLayout.addWidget(self.startPushButton)
        self.buttonWidget.setLayout(buttonLayout)

    def createTable(self):
        if os.path.isfile('saved.txt'):
            file = open('saved.txt', 'r')
            fileLength = len(file.readlines())
            if fileLength > 0:
                self.rows = fileLength
            file.close()
        else:
            self.rows = 0
        self.rows = 4   # da rimuovere, ora solo per provare funzionamento
        self.columns = 4
        self.startTable = QTableWidget(self.rows, self.columns)
        self.startTable.setHorizontalHeaderLabels(
            ['', 'Data', 'Performance', 'Parametri'])
        self.startTable.setColumnWidth(0, 30)
        self.startTable.setGeometry(300, 300, 250, 250)
        self.startTable.setDisabled(True)
        for row in range(self.rows):
            startRightCheckBoxItem = QTableWidgetItem(row)
            startRightCheckBoxItem.setFlags(
                Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
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
            values = re.findall(
                '[0-9]+', self.startTable.item(item.row(), 3).text())
            self.startLeftLineEdit1.setText(values[0])
            self.startLeftLineEdit2.setText(values[1])
            self.startLeftLineEdit3.setText(values[2])

    def loadDataset(self):
        options = QFileDialog().Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.startLeftDatasetLabel.setText(
                'Dataset caricato: \n' + os.path.basename(fileName))

    def runModel(self):  # TODO: to run model with inserted or selected parameters
        # add main of program
        pass

class StatisticPage(StartPage):
    def __init__(self):
        super().__init__()

        self.createAccuracyGraphWidget()
        self.createPrecisionGraphWidget()
        self.createFScoreGraphWidget()
        self.createRecallGraphWidget()
        self.createReturnButton()

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

    def createReturnButton(self):
        self.returnWidget = QWidget()

        self.returnButton = QPushButton()
        self.returnButton.setText('Torna alla pagina iniziale')
        self.returnButton.setFixedSize(200, 30)
        #TODO: function to connect to return to initial page
        returnLayout = QVBoxLayout()
        returnLayout.addWidget(self.returnButton)
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

class ResultPage(StartPage):
    def __init__(self):
        super().__init__()

        self.createResultWidget()
        self.createSaveWidget()
        self.returnStartPageWidget()

        resultPageLayout = QVBoxLayout()
        resultPageLayout.addWidget(self.resultWidget)
        resultPageLayout.addWidget(self.saveWidget)
        resultPageLayout.addWidget(self.returnWidget)
        self.setLayout(resultPageLayout)

    def createResultWidget(self):
        self.resultWidget = QWidget()

        self.resultLabel1 = Label('Risultato del calcolo sul modello configurato:')
        self.resultLabel2 = Label('Accuracy: ')
        self.resultLabel3 = Label('Precision: ')
        self.resultLabel4 = Label('Fscore: ')
        self.resultLabel5 = Label('Recall: ')

        resultLayout = QVBoxLayout()
        resultLayout.addWidget(self.resultLabel1)
        resultLayout.addWidget(self.resultLabel2)
        resultLayout.addWidget(self.resultLabel3)
        resultLayout.addWidget(self.resultLabel4)
        resultLayout.addWidget(self.resultLabel5)
        self.resultWidget.setLayout(resultLayout)

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

        saveLayout = QHBoxLayout()
        saveLayout.addWidget(self.saveLabel1)
        saveLayout.addWidget(self.saveYesButton)
        saveLayout.addWidget(self.saveNoButton)
        saveLayout.addItem(self.saveHSpacer)
        self.saveWidget.setLayout(saveLayout)

    def returnStartPageWidget(self):
        self.returnWidget = QWidget()

        self.returnButton = QPushButton()
        self.returnButton.setText('Torna alla pagina iniziale')
        self.returnButton.setFixedSize(200, 30)
        self.returnVSpacer = QSpacerItem(0, 100, QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.returnButton.clicked.connect(self.returnStartPage)

        returnLayout = QVBoxLayout()
        returnLayout.addItem(self.returnVSpacer)
        returnLayout.addWidget(self.returnButton)
        self.returnWidget.setLayout(returnLayout)

    def saveConfiguration(self):
        saveDate = datetime.datetime.now()
        # data conf, performance(label,valore), parametri(label,valore)
        file = open('saved.txt', 'a')
        file.write(str(saveDate.year) + '-' + str(saveDate.month) +
                   '-' + str(saveDate.day) + '\t')
        file.write('(' + self.resultLabel2.text() + ',' + 'valore' + '),' +
                   '(' + self.resultLabel3.text() + ',' + 'valore' + '),' +
                   '(' + self.resultLabel4.text() + ',' + 'valore' + '),' +
                   '(' + self.resultLabel5.text() + ',' + 'valore' + ')\t')
        file.write('(' + self.startLeftLabel2.text() + ',' + self.startLeftLineEdit1.text() + '),' +
                   '(' + self.startLeftLabel3.text() + ',' + self.startLeftLineEdit2.text() + '),' +
                   '(' + self.startLeftLabel4.text() + ',' + self.startLeftLineEdit3.text() + ')\n')
        file.close()

    def returnStartPage(self):
        #TODO: return button to start page
        pass

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
