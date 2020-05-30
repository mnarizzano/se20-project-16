from FeatureExtractor import FeatureExtractor
from MetaExtractor import MetaExtractor
from Model import Model


class Engine(Model):
    def process(self):

        meta = MetaExtractor()
        meta.annotateConcepts()
        '''
        # FeatureExtraction
        feature = FeatureExtractor()
        feature.extractNouns()
        '''

    def plot(self):
        self.plotConcept('1745121')

    def plotConcept(self, conceptId):
        for concept in self.dataset:
            if concept.id == conceptId:
                print("concept content is: " + concept.content)