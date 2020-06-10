from FeatureExtractor import FeatureExtractor
from Model import Model


class Engine(Model):
    def process(self):
        # for the moment just featureExtraction
        feature = FeatureExtractor()
        feature.extractSentences()


    def plot(self):
        self.plotConcept('1745121')

    def plotConcept(self, conceptId):
        for concept in self.dataset:
            if concept.id == conceptId:
                print("concept content is: " + concept.content)