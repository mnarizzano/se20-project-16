from FeatureExtractor import FeatureExtractor
from Model import Model
import time
import os
from Settings import Settings

class Engine(Model):
    def process(self):
        # for the moment just featureExtraction
        feature = FeatureExtractor()
        start_time = time.time()
        feature.extractSentences()
        elapsed_time = time.time() - start_time
        Settings.logger.debug('Cache: ' + str(Settings.useCache and os.path.exists(Settings.conceptsPickle)) +
                              ", Annotation Elapsed time: " + str(elapsed_time))
        feature.extractNounsVerbs()


    def plot(self):
        self.plotConcept('1745121')

    def plotConcept(self, conceptId):
        for concept in self.dataset:
            if concept.id == conceptId:
                print("concept content is: " + concept.content)