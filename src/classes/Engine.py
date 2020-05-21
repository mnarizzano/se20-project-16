from FeatureExtractor import FeatureExtractor

class Engine:
    def process(self):
        # for the moment just featureExtraction
        feature = FeatureExtractor()
        feature.extractNouns()