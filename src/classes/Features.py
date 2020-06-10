
class Features:
    conllu = None
    annotatedSentences = None # This contains a list where each entry is an annotated sentence of the concept

    def __init__(self):
        self.sentences = []
        self.nounsList = []
        self.verbsList = []
        self.nounsSet = set()
        self.verbsSet = set()

    def get_numberOfSentences(self):
        return len(self.sentences)

    def get_annotatedSentences(self):
        return self.sentences