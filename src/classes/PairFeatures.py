import wikipediaapi
from Settings import Settings
from Model import Model
import numpy as np

class PairFeatures:
    """ Contains Features related to pairs of Concepts

    Contains feature for all Concepts pair in the form of a linkedList
    Also offers setters and getters for each of them given a triplet ConceptA, ConceptB and domain
    Exposes Flags methods to check if a given metric is already present in the List

    ...

    Attributes
    ----------
    pairFeatures : {concept_1: {concept_X: PairFeats, concept_Y: PairFeats, ...}, concept_2: {}, ....}
        Linked list containing a pairFeatures object for each concept pair

    """

    class PairFeats:
        """Contains feature for a single pair of Concepts

        ...

        Attributes
        ----------
        link : bool
            Tracks if a direct link exist between 2 Concepts
        containsTitle : bool
            Tracks if a Concept Text contains the title of the other one
        jaccardSimilarity : int
            Contains the jaccard similarity value between texts of 2 Concepts
        referenceDistance : int
            Contains the RefD metric value between texts of 2 Concepts
        LDACrossEntropy : int
            Contains the crossentropy calculated on the LDA vectors of 2 Concepts
        LDA_KLDivergence : int
            Contains the LDA KLDivergence value calculated on the LDA vectors of 2 Concepts

        """
        def __init__(self):
            self.link = 0
            self.containsTitle = 0
            self.jaccardSimilarity = 0
            self.referenceDistance = 0
            self.LDACrossEntropy = 0
            self.LDA_KLDivergence = 0

    def __init__(self):
        """Initializes the Linked List
        """
        self.pairFeatures = {self.keyOf(j): {self.keyOf(i): self.PairFeats() for i in Model.dataset} for j in Model.dataset}    # dict is faster

    def keyOf(self, concept):
        """Defines the Key to use when indexing the Linked List
        """
        return concept.id

    def addLink(self, conceptA, conceptB, value):
        """Sets a link between ConceptA and ConceptB
        """
        self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].link = value

    def linksLoaded(self):
        """Controls if links were already loaded

        This method return whether or not links have already been extracted.
        Since RefD is 0 at the beginning we must make sure at least one RefD value is != 0.
        Plus we check whether the field "totalIncomingLinks" exist or not

        Returns:
            bool
        """
        loadedRefD = (self.pairFeatures[Model.dataset[-1].id][Model.dataset[-1].id].link is not None and
                sum([sum([self.pairFeatures[a.id][b.id].link for a in Model.dataset]) for b
                     in Model.dataset]) > 0)
        loadedTotalInLinks = ('totalIncomingLinks' in Model.dataset[-1].features.__dict__.keys() and
                Model.dataset[-1].features.totalIncomingLinks is not None and
                sum([concept.features.totalIncomingLinks for concept in Model.dataset]) > 0)
        return loadedRefD and loadedTotalInLinks

    def setJaccardSimilarity(self, conceptA, conceptB, js):
        """Setter for the jaccard Similarity feature
        """
        self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].jaccardSimilarity = js
        
    def setReferenceDistance(self, conceptA, conceptB, dist):
        """Setter for the RefD feature
        """
        self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].referenceDistance = dist

    def setLDACrossEntropy(self, conceptA, conceptB, value):
        """Setter for the LDA Cross Entropy feature
        """
        self.pairFeatures[self.keyOf(conceptA)][self.keyOf(conceptB)].LDACrossEntropy = value

    def LDACrossEntropyLoaded(self):
        """Checks whether if the LDA Cross Entropy has been already calulated
        """
        return (self.pairFeatures[Model.dataset[-1].id][Model.dataset[-1].id].LDACrossEntropy is not None and
                sum([sum([self.pairFeatures[a.id][b.id].LDACrossEntropy for a in Model.dataset]) for b
                     in Model.dataset]) > 0)

    def RefDLoaded(self):
        """Checks whether if the RefD has been already calulated
        """
        return (self.pairFeatures[Model.dataset[-1].id][Model.dataset[-1].id].referenceDistance is not None and
                np.array([[self.pairFeatures[a.id][b.id].referenceDistance for a in Model.dataset] for b
                in Model.dataset]).std() > 0)

    def containsTitleLoaded(self):
        """Checks whether if the ContainsTitle has been already calulated
        """
        return ('containsTitle' in self.pairFeatures[Model.dataset[-1].id][Model.dataset[-1].id].__dict__.keys() and
                self.pairFeatures[Model.dataset[-1].id][Model.dataset[-1].id].containsTitle is not None and
                sum([sum([self.pairFeatures[a.id][b.id].containsTitle for a in Model.dataset]) for b
                     in Model.dataset]) > 0)

    def jaccardLoaded(self):
        """Checks whether if the jaccard Similarity has been already calulated
        """
        return (self.pairFeatures[Model.dataset[-1].id][Model.dataset[-1].id].jaccardSimilarity is not None and
                sum([sum([self.pairFeatures[a.id][b.id].jaccardSimilarity for a in Model.dataset]) for b
                     in Model.dataset]) > 0)

    def setLDA_KLDivergence(self, conceptA, conceptB, value):
        """Setter for the LDA KLDivergence
        """
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