from Settings import Settings
from Model import Model

class GraphMatrix:

    unknown = -1
    isPrereq = 1
    notPrereq = 0
    domains = {}

    def __init__(self, domains):
        # -1 if prereq not present (in dataset), 0 if prereq explicitly 0 (in dataset), 1 if prereq
        #self.matrix = [[{k: self.unknown for k in domains} for i in range(len(Model.dataset))] for j in range(len(Model.dataset))]
        self.matrix = {}
        self.domains = domains
        self.numberOfPrereqs = 0
        self.numberOfNonPrereqs = 0
        self.unknownPrereqs = 0


    def getStatistics(self):
        return [self.numberOfPrereqs, self.numberOfNonPrereqs, self.unknownPrereqs]

    def addPrereq(self, conceptA, conceptB, value, domain):
        if not self.matrix.__contains__(conceptA):
            self.matrix[conceptA] = {}
        if not self.matrix[conceptA].__contains__(conceptB):
            self.matrix[conceptA][conceptB] = {}

        if not self.matrix[conceptA][conceptB].__contains__(domain):
            if int(value) == 0:
                self.numberOfNonPrereqs += 1
            elif int(value) == 1:
                self.numberOfPrereqs += 1
            self.matrix[conceptA][conceptB][domain] = int(value)
            self.unknownPrereqs = len(Model.dataset) ** 2 - self.numberOfNonPrereqs - self.numberOfPrereqs
        else:
            if int(value) != self.matrix[conceptA][conceptB][domain]:
                Settings.logger.error("Pairs prerequisite relation is inconsistent!!")
                raise Exception("Dataset inconsistent, found different annotation for  '" + conceptA + ', ' + conceptB + "'")
            else:
                Settings.logger.debug(
                    "Dataset warning: found duplicated annotation for  '" + conceptA + ', ' + conceptB + "' in '"+domain+"'")

    def plotGraph(self):
        for row in range(len(self.matrix[:][0])):
            row2string = ''
            for col in range(len(self.matrix[0][:])):
                row2string = row2string + str(self.matrix[row][col]) + " "
            Settings.logger.debug(row2string + "\n")

    def getPrereqs(self):
        return [prereq for prereq in self.matrix]

    def getPostreqs(self, prereq):
        return [postreq for postreq in self.matrix[prereq]]

    def getDomains(self, prereq, postreq):
        return [domain for domain in self.matrix[prereq][postreq]]

    def plotPrereqs(self):
        Settings.logger.debug(
            "Total concepts: " + str(len(Model.dataset)) + ". Total possible Prerequisites: " + str(len(Model.dataset)**2))
        Settings.logger.debug(
            "Total prereqs: " + str(self.numberOfPrereqs) + ", " + str(self.numberOfPrereqs*100 / len(Model.dataset)**2) + "%")
        Settings.logger.debug(
            "Total NonPrereqs: " + str(self.numberOfNonPrereqs) + ", " + str(self.numberOfNonPrereqs*100 / len(Model.dataset)**2) + "%")
        Settings.logger.debug(
            "Total Unknowns: " + str(self.unknownPrereqs) + ", " + str(self.unknownPrereqs*100 / len(Model.dataset)**2) + "%")
        for prereq in self.getPrereqs():
            for postreq in self.getPostreqs(prereq):
                for domain in Model.desiredGraph.getDomains(prereq, postreq):
                    Settings.logger.debug("[" + domain + "]" + Model.dataset[Model.dataset.index(prereq)].title + ", " +
                                          Model.dataset[Model.dataset.index(postreq)].title + ", " +
                                          str(self.matrix[prereq][postreq][domain]))

    def getPrereq(self, conceptA, conceptB, domain):
        return self.matrix[conceptA][conceptB][domain]