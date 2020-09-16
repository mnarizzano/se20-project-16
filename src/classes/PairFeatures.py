import wikipediaapi
from Settings import Settings
from Model import Model


class PairFeatures:

    # pairFeatures = {{}} # private Matrix of PairFeats objects (1 for each concept pair) as Dictionary

    class PairFeats:
        def __init__(self):
            self.link = None
            self.jaccardSimilarity = 0
            self.referenceDistance = 0
            self.LDACrossEntropy = 0
            self.LDA_KLDivergence = 0

    def __init__(self):
        # set a matrix of PairFeats
        # self.pairFeatures = [[self.PairFeats() for i in range(len(Model.dataset))] for j in range(len(Model.dataset))] # list version is inefficient
        self.pairFeatures = {self.keyOf(j): {self.keyOf(i): self.PairFeats() for i in Model.dataset} for j in Model.dataset}    # dict is faster

    def keyOf(self, concept):
        return concept.id

    # SETTERS
    def addLink(self, conceptA, conceptB, value):
        # adds link from A to B, NOTE that rows are "referencing" while columns are "referred"
        self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].link = value

    def linksLoaded(self):
        return self.pairFeatures[Model.dataset[-1].id][Model.dataset[-1].id].link is not None

    def setJaccardSimilarity(self, conceptA, conceptB, js):
        # (jaccardSimilarity is symmetric)
        self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].jaccardSimilarity = js
        
    def setReferenceDistance(self, conceptA, conceptB, dist):
        self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].referenceDistance = dist

    def setLDACrossEntropy(self, conceptA, conceptB, value):
        self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].LDACrossEntropy = value

    def setLDA_KLDivergence(self, conceptA, conceptB, value):
        self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].LDA_KLDivergence = value

    # GETTERS
    def getRefDistance(self, conceptA, conceptB):
        return self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].referenceDistance

    def getJaccardSim(self, conceptA, conceptB):
        return self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].jaccardSimilarity

    def getLDACrossEntropy(self, conceptA, conceptB):
        return self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].LDACrossEntropy

    def getLDA_KLDivergence(self, conceptA, conceptB):
        return self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].LDA_KLDivergence