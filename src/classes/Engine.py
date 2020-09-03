from FeatureExtractor import FeatureExtractor
from Model import Model
import time
import os
from PairFeatures import PairFeatures
from MetaExtractor import MetaExtractor
from Settings import Settings
from PairFeatures import PairFeatures

import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def build_baseline_model(input_size, output_size):
    return baseline_model(input_size, output_size)

class Engine(Model):

    pairFeatures = None

    def __init__(self):
        self.pairFeatures = PairFeatures()


    def process(self):
        # initialize and pass my PairFeatures to the FeatureExtractor
        feature = FeatureExtractor(self.pairFeatures)
        # begin processing of single Features
        start_time = time.time()
        feature.extractSentences()
        elapsed_time = time.time() - start_time
        Settings.logger.debug('Cache: ' + str(Settings.useCache and os.path.exists(Settings.conceptsPickle)) +
                              ", Annotation Elapsed time: " + str(elapsed_time))
        feature.extractNounsVerbs()
        feature.LDA()   # TODO: let LDA call extractNounsVerbs?

        # begin processing of pair Features

        ## processing of meta features
        Settings.logger.info("Fetching Meta Info...")
        meta = MetaExtractor(self.pairFeatures)
        meta.annotateConcepts()
        meta.extractLinkConnections()
        ### example for RefD calculation
        '''
        # call to RefD calculation (calculated between pairs, should move for:for: in FeatureExtractor and have single method)
        # feature.refDistance() which calculates them all and then getter as feature.getRefDistance(ConceptA, ConceptB)
        for conceptA in Model.dataset:
            for conceptB in Model.dataset:
                if conceptA.title == 'Significatività' and conceptB.title == 'Outlier':     # wikipedia pages for test
                    feature.referenceDistance(conceptA, conceptB)
        '''
        ## processing of raw features
        feature.jaccardSimilarity()
        feature.LDACrossEntropy()

        # create and train net
        encoder = LabelEncoder()
        result_set = self.classifierFormatter(feature)
        encoder.fit(result_set['desired'])
        encoded_Y = encoder.transform(result_set['desired'])
        dummy_y = np_utils.to_categorical(encoded_Y)
        Settings.logger.info("Starting Network training...")
        Settings.logger.info("Number of features = Input Size = " + str(result_set['input_size']))
        Settings.logger.info("Number of classes = Output Size = " + str(result_set['output_size']))
        # define baseline model
        def baseline_model():
            input_size = 2
            output_size = 2
            # create model
            model = Sequential()
            model.add(Dense(8, input_dim=result_set['input_size'], activation='relu'))
            model.add(Dense(result_set['output_size'], activation='softmax'))  # 3 if accepted output is isPrereq/notPrereq/unknown
            # Compile model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # add other metrics
            return model

        estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=5, verbose=0)
        kfold = KFold(n_splits=3, shuffle=True)
        X = np.array(result_set['features'])
        results = cross_val_score(estimator, X, dummy_y, cv=kfold)
        print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    def plot(self):
        # TODO: trigger GUI.plot() here to plot results
        pass

    def plotConcept(self, conceptId):
        # TODO: move to GUI.plot() concept?
        pass

    def classifierFormatter(self, feature, undersampleBiggerClass=False, resampleSmallerClass=False):
        # check all concept pairs and return their features and their desired prerequisite label
        features = []
        desired = []
        classRatio = {}
        for conceptA in Model.dataset:
            for conceptB in Model.dataset:
                # Only consider known relations since % of unknown is > 90% and biases the system to always output "UNKNOWN"
                if Model.desiredGraph.getPrereq(conceptA, conceptB) != Model.desiredGraph.unknown:
                    if not classRatio.__contains__(Model.desiredGraph.getPrereq(conceptA, conceptB)):
                        classRatio[Model.desiredGraph.getPrereq(conceptA, conceptB)] = 0
                    classRatio[Model.desiredGraph.getPrereq(conceptA, conceptB)] += 1 # increase this class counter
                    features.append([feature.getJaccardSim(conceptA, conceptB)])
                    desired.append(Model.desiredGraph.getPrereq(conceptA, conceptB))

        # classRatio has same value as GraphMatrix.getStatistics()
        if resampleSmallerClass:
            pass
        number_of_classes = len(list(set(desired))) # = 2 if classes are isPrereq/notPrereq, 3 if Unknown is allowed
        # since output class from estimator is array_encoded of the label it has a dimension === to the number of different clases
        return {'features': features, "desired": desired, "input_size": len(features[0]), "output_size": number_of_classes}


