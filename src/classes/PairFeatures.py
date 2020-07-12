from Settings import Settings
from Model import Model

class PairFeatures:

    # features = {{}} # private Matrix of PairFeats objects (1 for each concept pair) as Dictionary

    class PairFeats:
        def __init__(self):
            self.link = 0
            self.jaccardSimilarity = 0


    def __init__(self):
        # set a matrix of features
        # self.features = [[self.PairFeats() for i in range(len(Model.dataset))] for j in range(len(Model.dataset))] # list version is inefficient
        self.features = {self.keyOf(j): {self.keyOf(i): self.PairFeats() for i in Model.dataset} for j in Model.dataset}    # dict is faster


    def keyOf(self, concept):   # if we want to change key we only need to change this
        return concept.id


    def addLink(self, conceptA, conceptB):
        # adds link from A to B, NOTE that rows are "referencing" while columns are "referred"
        self.PairFeats[self.keyOf(conceptA)][self.keyOf(conceptB)].link = 1

    def setJaccardSimilarity(self, conceptA, conceptB, js):
        # jaccardSimilarity is symmetric, Add it to both A->B and B->A
        self.features[self.keyOf(conceptA)][self.keyOf(conceptB)].jaccardSimilarity = js
        self.features[self.keyOf(conceptB)][self.keyOf(conceptA)].jaccardSimilarity = js