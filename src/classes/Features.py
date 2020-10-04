
class Features:

    def __init__(self):
        self.conllu = None
        self.annotatedSentences = None # This contains a list where each entry is an annotated sentence of the concept
        self.sentences = []
        self.nounsList = []
        self.verbsList = []
        self.nounsSet = set()
        self.verbsSet = set()
        self.nounsPlain = []
        self.verbsPlain = []
        self.ldaVector = []
        self.LDAEntropy = 0
        self.totalIncomingLinks = 0
        self.totalOutgoingLinks = 0

    def get_numberOfSentences(self):
        return len(self.sentences)

    def get_annotatedSentences(self):
        return self.sentences

    def get_LDAVector(self):
        return self.ldaVector