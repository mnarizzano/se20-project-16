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

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=1, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # add other metrics
    return model

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
        #meta.annotateConcepts()
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
        encoder.fit(self.classifierFormatter(feature)['desired'])
        encoded_Y = encoder.transform(self.classifierFormatter(feature)['desired'])
        dummy_y = np_utils.to_categorical(encoded_Y)
        Settings.logger.info("Starting Network training...")

        estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=5, verbose=0)
        kfold = KFold(n_splits=3, shuffle=True)
        X = np.array(self.classifierFormatter(feature)['features'])
        results = cross_val_score(estimator, X, dummy_y, cv=kfold)
        print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    def plot(self):
        # TODO: trigger GUI.plot() here to plot results
        pass

    def plotConcept(self, conceptId):
        # TODO: move to GUI.plot() concept?
        pass

    def classifierFormatter(self, feature):
        features = []
        desired = []
        for conceptA in Model.dataset:
            for conceptB in Model.dataset:
                features.append([feature.getJaccardSim(conceptA, conceptB)])
                desired.append(Model.desiredGraph.getPrereq(conceptA, conceptB))
        return {'features': features, "desired": desired}


