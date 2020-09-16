import random

from FeatureExtractor import FeatureExtractor
from Model import Model
import time
import os
from PairFeatures import PairFeatures
from MetaExtractor import MetaExtractor
from Settings import Settings
from PairFeatures import PairFeatures
import pickle

import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold


class Engine(Model):

    pairFeatures = None

    def __init__(self):
        if os.path.exists(Settings.pairFeaturesPickle):
            with open(Settings.pairFeaturesPickle, 'rb') as file:
                self.pairFeatures = pickle.load(file)
        else: self.pairFeatures = PairFeatures()


    def process(self, parser):
        # initialize and pass my PairFeatures to the FeatureExtractor
        feature = FeatureExtractor(self.pairFeatures)
        # begin processing of single Features
        Settings.logger.debug('Starting sencence extraction (might take a lot)...')
        start_time = time.time()
        feature.extractSentences()
        elapsed_time = time.time() - start_time
        Settings.logger.debug('Using Cache: ' + str(Settings.useCache and os.path.exists(Settings.conceptsPickle)) +
                              ", Annotation Elapsed time: " + str(elapsed_time))
        parser.cache()
        feature.extractNounsVerbs()
        feature.LDA()   # TODO: let LDA call extractNounsVerbs?
        parser.cache()

        # begin processing of pair Features

        ## processing of meta features
        Settings.logger.info("Fetching Meta Info...")
        meta = MetaExtractor(self.pairFeatures)
        meta.annotateConcepts()
        parser.cache()
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
        parser.cache()

        # create and train net
        encoder = LabelEncoder()
        result_set = self.classifierFormatter(feature)
        encoder.fit(result_set['desired'])
        encoded_Y = encoder.transform(result_set['desired'])    # from generic label to integer: ['a', 'a', 'b', 1, 1, 1, 1]->[1, 1, 2, 0, 0, 0, 0]
        dummy_y = np_utils.to_categorical(encoded_Y)
        # the above line goes from integer to oneshot array encoded: [1, 1, 2, 0, 0, 0, 0] ->
        '''
          [[0., 1., 0.],
           [0., 1., 0.],
           [0., 0., 1.],
           [1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.]]
        '''
        Settings.logger.info("Starting Network training...")
        Settings.logger.info("Number of features = Input Size = " + str(result_set['input_size']))
        Settings.logger.info("Number of classes = Output Size = " + str(result_set['output_size']))
        # define baseline model
        def baseline_model():
            # create model
            model = Sequential()
            model.add(Dense(20, input_dim=result_set['input_size'], activation='relu'))
            model.add(Dense(20, input_dim=result_set['input_size'], activation='relu'))
            model.add(Dense(result_set['output_size'], activation='softmax'))  # 3 if accepted output is isPrereq/notPrereq/unknown
            # Compile model
            # whats the impact of metrics or loss when this gets managed from KerasClassifier?
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # add other metrics
            return model

        # classWeight = {0: 1, 1: 2}    # penalize errors on class 0 more than on class 1 since class 0 is half the number of samples of those of class 1
        estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=5, verbose=0)
        kfold = KFold(n_splits=3, shuffle=True)
        X = np.array(result_set['features'])
        results = cross_val_score(estimator, X, dummy_y, n_jobs=-1, cv=kfold,  fit_params={'class_weight': result_set['class_weights']})
        print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std() * 2))

        scoring = ['accuracy', 'f1', 'precision_macro', 'recall_macro'] # difference between macro and not macro? f1 === f-score?
        scores = cross_validate(estimator, X, encoded_Y, n_jobs=-1, scoring=scoring, cv=kfold,  fit_params={'class_weight': result_set['class_weights']})
        print(sorted(scores.keys()))

    def plot(self):
        # TODO: trigger GUI.plot() here to plot results
        pass

    def plotConcept(self, conceptId):
        # TODO: move to GUI.plot() concept?
        pass

    def classifierFormatter(self, feature, dropBiggerClass=False, resampleSmallerClass=True):    # resample = True changes results: why? shouldn't wheights account for unbalanced classes?!?
        # check all concept pairs and return their features and their desired prerequisite label
        features = []
        desired = []
        classRatio = {}
        for conceptA in Model.dataset:
            for conceptB in Model.dataset:
                # Only consider known relations since % of unknown is > 90% and biases the system to always output "UNKNOWN"
                if Model.desiredGraph.getPrereq(conceptA, conceptB) != Model.desiredGraph.unknown:
                    if not classRatio.__contains__(int(Model.desiredGraph.getPrereq(conceptA, conceptB))):
                        classRatio[int(Model.desiredGraph.getPrereq(conceptA, conceptB))] = 0
                    classRatio[int(Model.desiredGraph.getPrereq(conceptA, conceptB))] += 1 # increase this class counter
                    # TODO define above a simple dictionary containing features we want to consider and automatically map it here
                    features.append([feature.getJaccardSim(conceptA, conceptB),
                                     feature.getRefDistance(conceptA, conceptB),
                                     feature.getLDACrossEntropy(conceptA, conceptB),
                                     feature.getLDA_KLDivergence(conceptA, conceptB),
                                     *conceptA.getFeatures().get_LDAVector(),   # spread operator: *['a', 'b'] = a, b
                                     *conceptB.getFeatures().get_LDAVector(),
                                     # int(Model.desiredGraph.getPrereq(conceptA, conceptB))    # this is cheating but the model is not giving 100% in this case. Classifier too much simple?
                    ])
                    # features.append([random.choice([0, 1])])  # only one, random features: should return performance = 50%
                    # features.append([int(Model.desiredGraph.getPrereq(conceptA, conceptB))])   # truth oracle, should return performance = 100%
                    desired.append(int(Model.desiredGraph.getPrereq(conceptA, conceptB)))

        # classRatio has same value as GraphMatrix.getStatistics()
        if resampleSmallerClass:
            minorClass = 0 if classRatio[0] - classRatio[1] < 0 else 1
            for i in range(abs(classRatio[0] - classRatio[1])):
                randomItem = random.choice(range(min(classRatio[0], classRatio[1])))
                while desired[randomItem] != minorClass:    # really, really, really dumb way to resample, TODO: keep classes separate
                    randomItem = random.choice(range(min(classRatio[0], classRatio[1])))
                desired.append(desired[randomItem])
                features.append(features[randomItem])
            if desired.count(1) != desired.count(0):
                raise Exception("Classes are not balanced, # of 0: " + str(desired.count(0)) + ", # of 1: " + str(desired.count(1)))
        # if dropBiggerClass:    # TODO now implementing class balancing through class_weights
        number_of_classes = len(list(set(desired))) # = 2 if classes are isPrereq/notPrereq, 3 if Unknown is allowed

        # different examples for class weight blancing

        #weights = {0: 1, 1: classRatio[0] / classRatio[1]}  # ratio between classes
        #weights = {0: classRatio[1], 1: classRatio[0]}      # opposite ratio: in the end ratios are the same as above
        weights = {0: 1/desired.count(0), 1: 1/desired.count(0)}  # inverse ratio: in the end ratios are the same as above
        #weights = {0: 1, 1: 1}  # should behave as if no weights were specified


        # since output class from estimator is array_encoded of the label it has a dimension === to the number of different clases
        return {'features': features, "desired": desired, "input_size": len(features[0]),
                "output_size": number_of_classes, 'class_weights': weights}


