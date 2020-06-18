from FeatureExtractor import FeatureExtractor
from Model import Model
import time
import os
from Settings import Settings

class Engine(Model):

    pairFeatures = None

    def __init__(self):
        self.pairFeatures = PairFeatures()

    def process(self):
    
        meta = MetaExtractor(self.pairFeatures)
        meta.annotateConcepts()
        meta.extractLinkConnections()

        feature = FeatureExtractor(self.pairFeatures)

        for conceptA in Model.dataset:
            for conceptB in Model.dataset:
                if conceptA.title == 'Significatività' and conceptB.title == 'Outlier':     # wikipedia pages for test
                    feature.referenceDistance(conceptA, conceptB)

        
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
