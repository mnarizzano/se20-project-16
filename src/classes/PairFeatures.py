from Settings import Settings
from Model import Model

class PairFeatures:
    links = [] # -1 if prereq not present (in dataset), 0 if prereq explicitly 0 (in dataset), 1 if prereq

    def __init__(self):
        self.links = [[-1 for i in range(len(Model.dataset))] for j in range(len(Model.dataset))]


    def addLink(self, conceptA, conceptB):
        # adds link from A to B, NOTE that rows are "referencing" while columns are "referred"
        row = Model.dataset.index(conceptA)
        col = Model.dataset.index(conceptB)
        self.matrix[row][col] = 1