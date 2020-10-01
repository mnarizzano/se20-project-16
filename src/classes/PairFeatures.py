import wikipediaapi
from Settings import Settings
from Model import Model


class PairFeatures:

    # pairFeatures = {{}} # private Matrix of PairFeats objects (1 for each concept pair) as Dictionary

    class PairFeats:
        def __init__(self):
            # generic check
            # sum([sum([self.pairFeatures.pairFeatures[a.id][b.id].LDA_KLDivergence for a in MyModel.dataset]) for b in MyModel.dataset])
            self.link = 0
            self.containsTitle = 0
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
        # Probably std() is a better check than sum() (if they're all the same value, even if |= 0, they're still useless)
        return (self.pairFeatures[Model.dataset[-1].id][Model.dataset[-1].id].link is not None and
                sum([sum([self.pairFeatures[a.id][b.id].link for a in Model.dataset]) for b
                     in Model.dataset]) > 0)

    def setJaccardSimilarity(self, conceptA, conceptB, js):
        # (jaccardSimilarity is symmetric)
        self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].jaccardSimilarity = js
        
    def setReferenceDistance(self, conceptA, conceptB, dist):
        self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].referenceDistance = dist

    def setLDACrossEntropy(self, conceptA, conceptB, value):
        self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].LDACrossEntropy = value

    def LDACrossEntropyLoaded(self):
        return (self.pairFeatures[Model.dataset[-1].id][Model.dataset[-1].id].LDACrossEntropy is not None and
                sum([sum([self.pairFeatures[a.id][b.id].LDACrossEntropy for a in Model.dataset]) for b
                     in Model.dataset]) > 0)

    def RefDLoaded(self):
        return (self.pairFeatures[Model.dataset[-1].id][Model.dataset[-1].id].referenceDistance is not None and
                sum([sum([self.pairFeatures[a.id][b.id].referenceDistance for a in Model.dataset]) for b
                     in Model.dataset]) > 0)

    def containsTitleLoaded(self):
        return ('containsTitle' in self.pairFeatures[Model.dataset[-1].id][Model.dataset[-1].id].__dict__.keys() and
                self.pairFeatures[Model.dataset[-1].id][Model.dataset[-1].id].containsTitle is not None and
                sum([sum([self.pairFeatures[a.id][b.id].containsTitle for a in Model.dataset]) for b
                     in Model.dataset]) > 0)

    def jaccardLoaded(self):
        return (self.pairFeatures[Model.dataset[-1].id][Model.dataset[-1].id].jaccardSimilarity is not None and
                sum([sum([self.pairFeatures[a.id][b.id].jaccardSimilarity for a in Model.dataset]) for b
                     in Model.dataset]) > 0)

    def setLDA_KLDivergence(self, conceptA, conceptB, value):
        self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].LDA_KLDivergence = value

    def setContainsTitle(self, conceptA, conceptB, value):
        self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].containsTitle = value

    # GETTERS
    def getRefDistance(self, conceptA, conceptB):
        return self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].referenceDistance

    def getJaccardSim(self, conceptA, conceptB):
        return self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].jaccardSimilarity

    def getLDACrossEntropy(self, conceptA, conceptB):
        return self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].LDACrossEntropy

    def getLDA_KLDivergence(self, conceptA, conceptB):
        return self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].LDA_KLDivergence

    def getLink(self, conceptA, conceptB):
        return self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].link

    def getContainsTitle(self, conceptA, conceptB):
        return self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].containsTitle