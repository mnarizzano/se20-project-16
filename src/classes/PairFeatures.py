import wikipediaapi
from Settings import Settings
from Model import Model


class PairFeatures:

    class PairFeats:
        def __init__(self):
            self.link = 0
            self.referenceDistance = 0

    def __init__(self):
        # set a matrix of features
        # self.features = [[self.PairFeats() for i in range(len(Model.dataset))] for j in range(len(Model.dataset))] # list version is inefficient
        self.features = {self.keyOf(j): {self.keyOf(i): self.PairFeats() for i in Model.dataset} for j in Model.dataset}    # dict is faster

    def keyOf(self, concept):
        return concept.id

    def addLink(self, conceptA, conceptB):
        # adds link from A to B, NOTE that rows are "referencing" while columns are "referred"
        self.features[self.keyOf(conceptA)][self.keyOf(conceptB)].link = 1

    def setReferenceDistance(self, conceptA, conceptB, dist):
        self.features[self.keyOf(conceptA)][self.keyOf(conceptB)].referenceDistance = dist
