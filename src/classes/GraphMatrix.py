class GraphMatrix:
    concepts = []
    matrix = [] # -1 if prereq not present (in dataset), 0 if prereq explicitly 0 (in dataset), 1 if prereq
    logger = None

    def __init__(self, concepts, logger):
        self.logger = logger
        for concept in concepts:
            self.concepts.append(concept.title)
        self.matrix = [[-1 for i in range(len(concepts))] for j in range(len(concepts))]


    def addPrereq(self, conceptA, conceptB, value):
        # adds A->B, NOTE that rows are prerequisites while columns are "postrequisites"
        row = self.concepts.index(conceptA)
        col = self.concepts.index(conceptB)
        self.matrix[row][col] = value

    def plotGraph(self):
        for row in range(len(self.matrix[:][0])):
            row2string = ''
            for col in range(len(self.matrix[0][:])):
                row2string = row2string + str(self.matrix[row][col]) + " "
            print(row2string + "\n")
