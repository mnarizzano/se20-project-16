class Features:
    """ Engine

    Container for all single Concept Features

    ...

    Attributes
    ----------
    conllu : Udpipe.Model
        contains the conllu for the text of the concept
    annotatedSentences : []
        Contains a list where each entry is an annotated sentence of the concept
    sentences : [string]
        One entry for each sentence in the Concept Text
    nounsList : [string]
        List of nouns present in the concept Text
    verbsList : [string]
        List of verbs present in the concept Text
    nounsSet : [string]
        Set containing nouns present in the Concept Text
    verbsSet : [string]
        Set containing verbs present in the Concept Text
    nounsPlain : [string]
        List containing the lemma of each noun in the Concept Text
    verbsPlain : [string]
        List containing the lemma of each verb in the Concept Text
    ldaVector : [int]
        LDA Vector of the Concept Text
    LDAEntropy : int
        Entropy of the previous ldaVector
    totalIncomingLinks : int
        How many wikipedia pages points to this concept
    totalOutgoingLinks : int
        How many wikipedia pages are referenced by this concept

    """
    def __init__(self):
        self.conllu = None
        self.annotatedSentences = None
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
        """Counts how many sentences are present in the concept
        """
        return len(self.sentences)

    def get_annotatedSentences(self):
        """Getter for the list of sentences
        """
        return self.sentences

    def get_LDAVector(self):
        """Getter for the LDA Vector of the Text of the Concept
        """
        return self.ldaVector