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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from Parser import Parser

class Engine(Model):

    pairFeatures = None
    parser = Parser()

    def __init__(self):
        if os.path.exists(Settings.pairFeaturesPickle):
            with open(Settings.pairFeaturesPickle, 'rb') as file:
                self.pairFeatures = pickle.load(file)
        else: self.pairFeatures = PairFeatures()


    def process(self):
        # initialize and pass my PairFeatures to the FeatureExtractor
        feature = FeatureExtractor(self.pairFeatures)
        # begin processing of single Features
        Settings.logger.debug('Starting sencence extraction (might take a lot)...')
        start_time = time.time()
        feature.extractSentences()
        elapsed_time = time.time() - start_time
        Settings.logger.debug('Using Cache: ' + str(Settings.useCache and os.path.exists(Settings.conceptsPickle)) +
                              ", Annotation Elapsed time: " + str(elapsed_time))
        feature.extractNounsVerbs()
        feature.LDA()   # TODO: let LDA call extractNounsVerbs?

        # begin processing of pair Features

        ## processing of meta features
        Settings.logger.info("Fetching Meta Info...")
        meta = MetaExtractor(self.pairFeatures)
        meta.annotateConcepts()
        meta.extractLinkConnections()
        meta.referenceDistance()
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

        # obtain input and output from desiredGraphMatrix, PairFeatures and Model.dataset (for single ones)
        encoder = LabelEncoder()
        result_set = self.classifierFormatter()
        # featurs = inputs
        X = np.array(result_set['features'])
        # labels = outputs
        encoder.fit(result_set['desired'])
        encoded_Y = encoder.transform(result_set['desired'])    # from generic label to integer: ['a', 'a', 'b', 1, 1, 1, 1]->[1, 1, 2, 0, 0, 0, 0]
        # next line goes from integer to oneshot array encoded: [1, 1, 2, 0, 0, 0, 0] ->
        '''
          [[0., 1., 0.],
           [0., 1., 0.],
           [0., 0., 1.],
           [1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.]]
        '''
        dummy_y = np_utils.to_categorical(encoded_Y)

        # Log data and build model with input and output size based on data's ones
        Settings.logger.info("Starting Network training...")
        Settings.logger.debug("Number of features = Input Size = " + str(result_set['input_size']))
        modelOutput = result_set['output_size']
        activationFunction = 'softmax'
        lossFunction = 'categorical_crossentropy'
        if modelOutput == 2:
            modelOutput = 1  # for binary classification 2 classes are classified by 1 probability, wile x classes (>2) are classified by x-1 probabilities
            activationFunction = 'sigmoid'
            lossFunction = 'binary_crossentropy'
        Settings.logger.info("Number of classes = " + str(result_set['output_size']))
        Settings.logger.info("Output Size = " + str(modelOutput))
        Settings.logger.info("Last layer activation function = " + activationFunction)
        # define neural network
        def neural_network():
            # create model
            model = Sequential()
            model.add(Dense(20, input_dim=result_set['input_size'], activation='relu'))
            # TODO: might add drop layer to mitigate overfitting
            model.add(Dense(20, activation='relu'))
            model.add(Dense(modelOutput, activation=activationFunction))  # 3 if accepted output is isPrereq/notPrereq/unknown
            # Compile model
            # whats the impact of metrics or loss when this gets managed from KerasClassifier?
            model.compile(loss=lossFunction, optimizer='adam', metrics=['accuracy'])  # add other metrics?
            return model

        # classWeight = {0: 1, 1: 2}    # penalize errors on class 0 more than on class 1 since class 0 is half the number of samples of those of class 1
        estimator = KerasClassifier(build_fn=neural_network, epochs=20, batch_size=5, verbose=0)
        estimator._estimator_type = "classifier"

        if not Settings.generateOutput: # split train dataset for Cross-validation run
            # kfold = KFold(n_splits=10, shuffle=True)
            # StratifiedKFold tries to balance set classes between Folds, 7 is a random number not set to random just for reproducibility
            # kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=7)
            numberOfSplits = 3
            kfold = StratifiedShuffleSplit(n_splits=numberOfSplits, test_size=1/numberOfSplits)
            for train, test in kfold.split(X, encoded_Y):
                Settings.logger.debug('train -  {}   |   test -  {}'.format(
                np.bincount(encoded_Y[train]), np.bincount(encoded_Y[test])))
            #results = cross_val_score(estimator, X, dummy_y, n_jobs=-1, cv=kfold,  fit_params={'class_weight': result_set['class_weights']})
            #print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std() * 2))

            scoring = ['accuracy', 'f1_macro', 'f1_micro', 'average_precision', 'balanced_accuracy', 'precision_macro', 'recall_macro'] # difference between macro and not macro? f1 === f-score?
            scores = cross_validate(estimator, X, encoded_Y, n_jobs=-1, scoring=scoring, cv=kfold, fit_params={'class_weight': result_set['class_weights']})
            #scores = cross_validate(neural_network(), X, encoded_Y, n_jobs=-1, scoring=scoring, cv=kfold,
            #                        fit_params={'class_weight': result_set['class_weights'], 'epochs':20, 'batch_size':5, 'verbose':0})
            Settings.logger.debug(str(scores))
            Settings.logger.debug("Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
            Settings.logger.debug("Recall: %0.2f (+/- %0.2f)" % (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std() * 2))
            Settings.logger.debug("Precision: %0.2f (+/- %0.2f)" % (scores['test_average_precision'].mean(), scores['test_average_precision'].std() * 2))
            Settings.logger.debug("F1: %0.2f (+/- %0.2f)" % (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std() * 2))

            i = 1
            for train_index, test_index in kfold.split(X, encoded_Y):
                X_train = X[train_index]
                X_test = X[test_index]
                y_train = encoded_Y[train_index]
                y_test = encoded_Y[test_index]

                model = neural_network()
                #model = KerasClassifier(build_fn=neural_network, epochs=20, batch_size=5, verbose=0)
                #model._estimator_type = "classifier"

                # Train the model
                model.fit(X_train, y_train, epochs=20, batch_size=5)  # Training the model
                print(f"Accuracy for the fold no. {i} on the test set: {accuracy_score(y_test,(model.predict(X_test) > 0.5).astype('int32'))}")
                print("accuracy as seen from model.evaluate: " + str(model.evaluate(X_test, y_test)[1]))  # if plain simple keras sequential model
                #print("accuracy as seen from model.evaluate: " + str(model.score(X_test, y_test)))          # if KerasClassifier wrapper
                i += 1

            return ({'accuracy': scores['test_accuracy'].mean(), 'recall': scores['test_recall_macro'].mean(),
                     'precision': scores['test_average_precision'].mean(), 'f1': scores['test_f1_macro'].mean()})
        else:   # we're using the network to predict labels, not cross-validate, get predictions for test set
            model = KerasClassifier(build_fn=neural_network, epochs=20, batch_size=5, verbose=0)
            Settings.logger.debug('Started training...')
            model.fit(X, encoded_Y)
            output = {}
            Settings.logger.debug('Started prediction...')
            for domain in self.parser.test:
                output[domain] = []
                for pair in self.parser.test[domain]:
                    fromConcept = Model.dataset[Model.dataset.index(pair[0])]
                    toConcept = Model.dataset[Model.dataset.index(pair[1])]
                    result = [fromConcept.title, toConcept.title, (model.predict(np.array([self.getFeatures(fromConcept, toConcept)])) > 0.5).astype('int32')]
                    output[domain].append(result)
            Settings.logger.debug('Found ' + str(sum([sum([pair[2] for pair in output[domain]]) for domain in output])) +
                                                 ' prereqs in a ' + str(sum([len([pair[2] for pair in output[domain]]) for domain in output])) + ' long testSet')
            return result

    def plot(self):
        # TODO: trigger GUI.plot() here to plot results
        pass

    def plotConcept(self, conceptId):
        # TODO: move to GUI.plot() concept?
        pass

    def classifierFormatter(self):    # resample = True changes results: why? shouldn't wheights account for unbalanced classes?!?
        Settings.logger.debug('Beginning dataset formatting')
        # check all concept pairs and return their features and their desired prerequisite label
        prereqData = []
        notPrereqData = []
        prereqLabel = []
        notPrereqLabel = []
        total = 0
        classRatio = {}
        for conceptA in Model.dataset:
            for conceptB in Model.dataset:
                # Only consider known relations since % of unknown is > 90% and biases the system to always output "UNKNOWN"
                if Model.desiredGraph.getPrereq(conceptA, conceptB) != Model.desiredGraph.unknown:  # spends a lot of time here, linked list would be better
                    total = total+1
                    # counter: counts every class occurrencies, creates new class if it hasn't yet encountered it
                    if not classRatio.__contains__(int(Model.desiredGraph.getPrereq(conceptA, conceptB))):
                        classRatio[int(Model.desiredGraph.getPrereq(conceptA, conceptB))] = 0
                    classRatio[int(Model.desiredGraph.getPrereq(conceptA, conceptB))] += 1 # increase this class counter
                    # TODO define above a simple dictionary containing features we want to consider and automatically map it here
                    feat = self.getFeatures(conceptA, conceptB)
                    # feat = [random.choice([0, 1])]  # only one, random features: should return performance = 50%
                    # feat = [int(Model.desiredGraph.getPrereq(conceptA, conceptB))]   # truth oracle, should return performance = 100%

                    if int(Model.desiredGraph.getPrereq(conceptA, conceptB)) == Model.desiredGraph.isPrereq:
                        prereqData.append(feat)
                        prereqLabel.append(int(Model.desiredGraph.getPrereq(conceptA, conceptB)))

                    if int(Model.desiredGraph.getPrereq(conceptA, conceptB)) == Model.desiredGraph.notPrereq:
                        notPrereqData.append(feat)
                        notPrereqLabel.append(int(Model.desiredGraph.getPrereq(conceptA, conceptB)))
        if len(notPrereqLabel) + len(prereqLabel) != total:
            raise Exception("Not all labels are of prerequisition")
        if abs(classRatio[0] - classRatio[1]) != abs(len(notPrereqLabel) - len(prereqLabel)):
            raise Exception("Something wrong in classes count")
        # classRatio has same value as GraphMatrix.getStatistics()
        minorData = notPrereqData if len(notPrereqLabel) - len(prereqLabel) < 0 else prereqData
        biggerData = prereqData if len(notPrereqLabel) - len(prereqLabel) < 0 else notPrereqData

        minorLabel = notPrereqLabel if len(notPrereqLabel) - len(prereqLabel) < 0 else prereqLabel
        biggerLabel = prereqLabel if len(notPrereqLabel) - len(prereqLabel) < 0 else notPrereqLabel
        pickedIndex = []
        if Settings.resampleSmallerClass:
            while abs(len(minorLabel)-len(biggerLabel)) > 0:
                randomItem = random.choice(range(len(minorLabel)))
                if not pickedIndex.__contains__(randomItem):
                    pickedIndex.append(randomItem)
                    minorLabel.append(minorLabel[randomItem])
                    minorData.append(minorData[randomItem])
            Settings.logger.debug("resampled a total of " + str(len(pickedIndex)) + " concepts")
        else:
            while abs(len(minorLabel) - len(biggerLabel)) > 0:
                randomItem = random.choice(range(len(biggerLabel)))
                pickedIndex.append(randomItem)
                biggerLabel.pop(randomItem)
                biggerData.pop(randomItem)
            Settings.logger.debug("dropped a total of " + str(len(pickedIndex)) + " concepts")

        if len(pickedIndex) != abs(classRatio[0] - classRatio[1]):
            raise Exception("Something wrong resampling lowerClass: resampled " + str(
                len(pickedIndex)) + ", original difference " + str(abs(classRatio[0] - classRatio[1])))

        number_of_classes = 2 # 2 if classes are isPrereq/notPrereq, 3 if Unknown is allowed

        # different examples for class weight blancing

        #weights = {0: 1, 1: classRatio[0] / classRatio[1]}  # ratio between classes
        #weights = {0: classRatio[1], 1: classRatio[0]}      # opposite ratio: in the end ratios are the same as above
        weights = {0: 1/len(prereqLabel), 1: 1/len(notPrereqLabel)}  # inverse ratio: in the end ratios are the same as above
        #weights = {0: 1, 1: 1}  # should behave as if no weights were specified


        features = [*prereqData, *notPrereqData]
        labels = [*prereqLabel, *notPrereqLabel]
        Settings.logger.debug('Finished dataset formatting')
        # since output class from estimator is array_encoded of the label it has a dimension === to the number of different clases
        return {'features': features, "desired": labels, "input_size": len(features[0]),
                "output_size": number_of_classes, 'class_weights': weights}

    def getFeatures(self, conceptA, conceptB):
        return [
            self.pairFeatures.getJaccardSim(conceptA, conceptB),
            self.pairFeatures.getLink(conceptA, conceptB),
            self.pairFeatures.getRefDistance(conceptA, conceptB),
            self.pairFeatures.getLDACrossEntropy(conceptA, conceptB),
            self.pairFeatures.getLDA_KLDivergence(conceptA, conceptB),
            *conceptA.getFeatures().get_LDAVector(),  # spread operator: *['a', 'b'] = a, b
            *conceptB.getFeatures().get_LDAVector(),
            # int(Model.desiredGraph.getPrereq(conceptA, conceptB))    # this is cheating but the model is not giving 100% in this case. Classifier too much simple?
         ]
