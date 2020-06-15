from FeatureExtractor import FeatureExtractor
from Model import Model
import time
import os
from Settings import Settings
from PairFeatures import PairFeatures
class Engine(Model):

    pairFeatures = None

    def __init__(self):
        self.pairFeatures = PairFeatures()


    def process(self):
        # initialize and pass my PairFeatures to the FeatureExtractor
        feature = FeatureExtractor(self.pairFeatures)

        # processing of single Features
        start_time = time.time()
        feature.extractSentences()
        elapsed_time = time.time() - start_time
        Settings.logger.debug('Cache: ' + str(Settings.useCache and os.path.exists(Settings.conceptsPickle)) +
                              ", Annotation Elapsed time: " + str(elapsed_time))
        feature.extractNounsVerbs()

        # begin processing of pair Features
        feature.jaccardSimilarity()



    def plot(self):
        self.plotConcept('1745121')

    def plotConcept(self, conceptId):
        for concept in self.dataset:
            if concept.id == conceptId:
                print("concept content is: " + concept.content)