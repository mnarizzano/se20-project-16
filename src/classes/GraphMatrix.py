from Settings import Settings
from Model import Model

class GraphMatrix:
    matrix = [] # -1 if prereq not present (in dataset), 0 if prereq explicitly 0 (in dataset), 1 if prereq

    def __init__(self):
        self.matrix = [[-1 for i in range(len(Model.dataset))] for j in range(len(Model.dataset))]


    def addPrereq(self, conceptA, conceptB, value):
        # adds A->B, NOTE that rows are prerequisites while columns are "postrequisites"
        row = Model.dataset.index(conceptA)
        col = Model.dataset.index(conceptB)
        self.matrix[row][col] = value

    def plotGraph(self):
        for row in range(len(self.matrix[:][0])):
            row2string = ''
            for col in range(len(self.matrix[0][:])):
                row2string = row2string + str(self.matrix[row][col]) + " "
            Settings.logger.debug(row2string + "\n")

    def plotPrereqs(self):
        for row in range(len(self.matrix[:][0])):
            for col in range(len(self.matrix[0][:])):
                if self.matrix[row][col] != -1:
                    Settings.logger.debug("[" + Model.dataset[row].domain + "]" + Model.dataset[row].title + ", " +
                                          "[" + Model.dataset[col].domain + "]" + Model.dataset[col].title + ", " +
                                          str(self.matrix[row][col]))

    def getPrereq(self, conceptA, conceptB):
        # adds A->B, NOTE that rows are prerequisites while columns are "postrequisites"
        row = Model.dataset.index(conceptA.title)
        col = Model.dataset.index(conceptB.title)
        return self.matrix[row][col]