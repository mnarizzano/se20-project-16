from Settings import Settings
from Model import Model

class GraphMatrix:

    unknown = -1
    isPrereq = 1
    notPrereq = 0
    def __init__(self):
        # -1 if prereq not present (in dataset), 0 if prereq explicitly 0 (in dataset), 1 if prereq
        self.matrix = [[self.unknown for i in range(len(Model.dataset))] for j in range(len(Model.dataset))]
        self.numberOfPrereqs = 0
        self.numberOfNonPrereqs = 0
        self.unknownPrereqs = 0


    def getStatistics(self):
        return [self.numberOfPrereqs, self.numberOfNonPrereqs, self.unknownPrereqs]

    def addPrereq(self, conceptA, conceptB, value):
        # adds A->B, NOTE that rows are prerequisites while columns are "postrequisites"
        row = Model.dataset.index(conceptA)
        col = Model.dataset.index(conceptB)
        self.matrix[row][col] = value
        if int(value) == 0:
            self.numberOfNonPrereqs += 1
        elif int(value) == 1:
            self.numberOfPrereqs += 1
        self.unknownPrereqs = len(Model.dataset) ** 2 - self.numberOfNonPrereqs - self.numberOfPrereqs

    def plotGraph(self):
        for row in range(len(self.matrix[:][0])):
            row2string = ''
            for col in range(len(self.matrix[0][:])):
                row2string = row2string + str(self.matrix[row][col]) + " "
            Settings.logger.debug(row2string + "\n")

    def plotPrereqs(self):
        Settings.logger.debug(
            "Total concepts: " + str(len(Model.dataset)) + ". Total possible Prerequisites: " + str(len(Model.dataset)**2))
        Settings.logger.debug(
            "Total prereqs: " + str(self.numberOfPrereqs) + ", " + str(self.numberOfPrereqs*100 / len(Model.dataset)**2) + "%")
        Settings.logger.debug(
            "Total NonPrereqs: " + str(self.numberOfNonPrereqs) + ", " + str(self.numberOfNonPrereqs*100 / len(Model.dataset)**2) + "%")
        Settings.logger.debug(
            "Total Unknowns: " + str(self.unknownPrereqs) + ", " + str(self.unknownPrereqs*100 / len(Model.dataset)**2) + "%")
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