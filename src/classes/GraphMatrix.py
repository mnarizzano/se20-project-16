__author__ = "Moggio Alessio, Parizzi Andrea"
__license__ = "Public Domain"
__version__ = "1.0"

from Settings import Settings
from Model import Model

class GraphMatrix:
    """ GraphMatrix

    Linked List of labels between pairs of Concepts

    ...

    Attributes
    ----------
    unknown : int
        Define the value used to express an unknown prerequisite relation
    isPrereq : int
        Define the value used to express a known prerequisite
    notPrereq : int
        Define the value used to express the known absence of a prerequisite
    matrix : {concept_1: {concept_X: bool, concept_Y: bool, ...}, concept_2: {}, ....}
        Linked list wich tracks prereq existence between pairs of Concepts
    domains : [string]
        tracks what are the domain of the run
    numberOfPrereqs : int
        Counter that tracks how many prereq relations exist
    numberOfNonPrereqs : int
        Counter that tracks how many non-prereq relations exist
    unknownPrereqs : int
        Counter that tracks how many unknown prereq relations exist
    """
    unknown = -1
    isPrereq = 1
    notPrereq = 0
    matrix = {}
    domains = {}
    numberOfPrereqs = 0
    numberOfNonPrereqs = 0
    unknownPrereqs = 0

    def __init__(self, domains):
        self.domains = domains

    def getStatistics(self):
        return [self.numberOfPrereqs, self.numberOfNonPrereqs, self.unknownPrereqs]

    def addPrereq(self, conceptA, conceptB, value, domain):
        """Given a pair of Concepts, their domain and the prereq
           relation between them adds it to the LinkedList
        """
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
        """Plots the prereq List in matrix Form
        """
        for row in range(len(self.matrix[:][0])):
            row2string = ''
            for col in range(len(self.matrix[0][:])):
                row2string = row2string + str(self.matrix[row][col]) + " "
            Settings.logger.debug(row2string + "\n")

    def getPrereqs(self):
        """Returns all prereqs in ListForm
        """
        return [prereq for prereq in self.matrix]

    def getPostreqs(self, prereq):
        """Given a pre-Concepts returns all post-Concepts that rely on it
        """
        return [postreq for postreq in self.matrix[prereq]]

    def getDomains(self, prereq, postreq):
        """Returns a list of all the Domains present
        """
        return [domain for domain in self.matrix[prereq][postreq]]

    def plotPrereqs(self):
        """Console log all prereq relations
        """
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
        """Given a pair of concepts and their domain return the prereq relation between them
        """
        return self.matrix[conceptA][conceptB][domain]