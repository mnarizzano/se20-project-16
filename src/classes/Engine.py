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
from keras.metrics import FalseNegatives, TrueNegatives, FalsePositives, TruePositives, Precision, Recall, Accuracy
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from Parser import Parser

class Engine(Model):

    pairFeatures = None
    parser = Parser()
    inputSize = None
    outputSize = None
    groups = None
    labels = None
    inputs = None
    classifier = None
    weights = None
    accuracy = None
    recall = None
    precision = None
    fscore = None
    network = None

    def __init__(self):
        if os.path.exists(Settings.pairFeaturesPickle):
            with open(Settings.pairFeaturesPickle, 'rb') as file:
                self.pairFeatures = pickle.load(file)
        else: self.pairFeatures = PairFeatures()


    def calculateFeatures(self):
        ## processing of raw features
        # begin processing of single Features
        feature = FeatureExtractor(self.pairFeatures)
        Settings.logger.debug('Starting sencence extraction (might take a lot)...')
        start_time = time.time()
        feature.extractSentences()
        elapsed_time = time.time() - start_time
        Settings.logger.debug('Using Cache: ' + str(Settings.useCache and os.path.exists(Settings.conceptsPickle)) +
                              ", Annotation Elapsed time: " + str(elapsed_time))
        feature.extractNounsVerbs()
        feature.LDA()  # let LDA call extractNounsVerbs?
        feature.containsTitle()

        # begin processing of pair Features
        feature.jaccardSimilarity()
        feature.LDACrossEntropy()

        ## processing of meta features
        Settings.logger.info("Fetching Meta Info...")
        meta = MetaExtractor(self.pairFeatures)
        meta.annotateConcepts()
        meta.extractLinkConnections()
        meta.referenceDistance()

    def encodeInputOutputs(self):
        # get and encode features and labels from train set for CV and for obtaining results
        encoder = LabelEncoder()
        # obtain input and output from desiredGraphMatrix, PairFeatures and Model.dataset (for single ones)
        result_set = self.classifierFormatter()
        x = np.array(result_set['features'])
        encoder.fit(result_set['desired'])
        encoded_y = encoder.transform(
            result_set['desired'])  # from generic label to integer: ['a', 'a', 'b', 1, 1, 1, 1]->[1, 1, 2, 0, 0, 0, 0]

        '''
        # next line goes from integer to oneshot array encoded: [1, 1, 2, 0, 0, 0, 0] ->
          [[0., 1., 0.],
           [0., 1., 0.],
           [0., 0., 1.],
           [1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.]]
        dummy_y = np_utils.to_categorical(encoded_y)
        '''

        self.labels = encoded_y
        self.inputs = x

    def buildNetwork(self):
        activationFunction = 'softmax'
        lossFunction = 'categorical_crossentropy'
        if self.outputSize == 2 or self.outputSize == 1:
            self.outputSize = 1  # for binary classification 2 classes are classified by 1 probability, wile x classes (>2) are classified by x-1 probabilities
            activationFunction = 'sigmoid'
            lossFunction = 'binary_crossentropy'
        Settings.logger.debug("Number of features = Input Size = " + str(self.inputSize))
        Settings.logger.info("Number of classes = Output Size = " + str(self.outputSize))
        Settings.logger.info("Last layer activation function = " + activationFunction)

        # define neural network
        def neural_network():
            # create model
            model = Sequential()
            # build model with input and output size based on data's ones
            model.add(Dense(Settings.neurons, input_dim=self.inputSize, activation='relu'))
            for i in range(int(Settings.layers)):
                # TODO: might add drop layer to mitigate overfitting
                model.add(Dense(int(Settings.neurons), activation='relu'))
            model.add(
                Dense(self.outputSize, activation=activationFunction))
            # Compile model
            # whats the impact of metrics or loss when this gets managed from KerasClassifier?
            metrics = ['accuracy', 'binary_accuracy'] #, TruePositives(), FalsePositives(), TrueNegatives(),
                       # FalseNegatives(), Precision(), Recall()]
            model.compile(loss=lossFunction, optimizer='adam', metrics=metrics)  # add other metrics?
            return model

        # classWeight = {0: 1, 1: 2}    # penalize errors on class 0 more than on class 1 since class 0 is half the number of samples of those of class 1
        estimator = KerasClassifier(build_fn=neural_network, epochs=int(Settings.epoch), batch_size=5, verbose=0)
        estimator._estimator_type = "classifier"
        self.classifier = estimator
        self.network = neural_network

    def autoCV(self):
        self.buildNetwork()
        # TODO: understand differences between these scoring variants
        # macro: calculates independently for each class and then average
        # micro: will calculates metrics sample by sample (not clustering by class) -> this is the 1 we want
        # accuracy is global: total % of correctly predicted labels
        # balanced_accuracy is % of correct per class, then averaged together
        # Precision refers to precision at a particular decision threshold. For example, if you count any model output less than 0.5 as negative, and greater than 0.5 as positive. But sometimes (especially if your classes are not balanced, or if you want to favor precision over recall or vice versa), you may want to vary this threshold. Average precision gives you average precision at all such possible thresholds, which is also similar to the area under the precision-recall curve. It is a useful metric to compare how well models are ordering the predictions, without considering any specific decision threshold.
        scoring = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_micro', 'average_precision', 'precision_micro',
                   'recall_micro']  # difference between macro and not macro? f1 === f-score?
        if not Settings.CrossDomain:    # train on all domains->use stratifiedKFold
            Settings.logger.debug('In-domain cross-validation')
            # kfold = KFold(n_splits=10, shuffle=True)
            # StratifiedKFold tries to balance set classes between Folds, 7 is a random number not set to random just for reproducibility
            # kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=7)
            kfold = StratifiedShuffleSplit(n_splits=int(Settings.kfoldSplits), test_size=1/int(Settings.kfoldSplits))
            # Show distribution of cross split for debug
            for train, test in kfold.split(self.inputs, self.labels):
                Settings.logger.debug('train -  {}   |   test -  {}'.format(
                np.bincount(self.labels[train]), np.bincount(self.labels[test])))
            # Actually perform CV
            scores = cross_validate(self.classifier, self.inputs, self.labels, n_jobs=-1, scoring=scoring, cv=kfold,
                                    fit_params={'class_weight': self.weights})
        else: # Use LeaveOneGroupOut KFold:
            Settings.logger.debug('Cross-domain cross-validation')
            kfold = LeaveOneGroupOut()
            # StratifiedShuffleSplit(n_splits=int(Settings.kfoldSplits), test_size=1/int(Settings.kfoldSplits))
            # Show distribution of cross split
            for train, test in kfold.split(self.inputs, self.labels, groups=self.groups):
                Settings.logger.debug('train -  {}   |   test -  {}'.format(
                np.bincount(self.labels[train]), np.bincount(self.labels[test])))
            # Actually perform CV
            scores = cross_validate(self.classifier, self.inputs, self.labels, n_jobs=-1, scoring=scoring, cv=kfold,
                                    groups=self.groups, fit_params={'class_weight': self.weights})

        # cross_val_score is very limited on which performances it calculates
        # results = cross_val_score(estimator, X, dummy_y, n_jobs=-1, cv=kfold,  fit_params={'class_weight': result_set['class_weights']})
        # print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std() * 2))

        # this is the call to make if, instead of wrapping NN in Keras classifier, we use neural_network() directly
        #scores = cross_validate(neural_network(), X, encoded_Y, n_jobs=-1, scoring=scoring, cv=kfold,
        #                        fit_params={'class_weight': result_set['class_weights'], 'epochs':int(Settings.epoch), 'batch_size':5, 'verbose':0})

        Settings.logger.debug('cross_validate CV performances: ' + str(scores))
        Settings.logger.debug("Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
        Settings.logger.debug("Recall: %0.2f (+/- %0.2f)" % (scores['test_recall_micro'].mean(), scores['test_recall_micro'].std() * 2))
        Settings.logger.debug("Precision: %0.2f (+/- %0.2f)" % (scores['test_precision_micro'].mean(), scores['test_precision_micro'].std() * 2))
        Settings.logger.debug("F1: %0.2f (+/- %0.2f)" % (scores['test_f1_micro'].mean(), scores['test_f1_micro'].std() * 2))

        self.accuracy = {'mean': scores['test_accuracy'].mean(), 'std': scores['test_accuracy'].std() * 2}
        self.recall = {'mean': scores['test_recall_micro'].mean(), 'std': scores['test_recall_micro'].std() * 2}
        self.precision = {'mean': scores['test_precision_micro'].mean(), 'std': scores['test_precision_micro'].std() * 2}
        self.fscore = {'mean': scores['test_f1_micro'].mean(), 'std': scores['test_f1_micro'].std() * 2}
        # destroy trained model to avoid interfering with other CV or prediction
        self.network = None
        self.classifier = None

    def manualCV(self):
        # Manual CV to enter in the loop and extract some insight
        if not Settings.CrossDomain:  # train on all domains->use stratifiedKFold
            Settings.logger.debug('In-domain manual CV')
            kfold = StratifiedShuffleSplit(n_splits=int(Settings.kfoldSplits), test_size=1 / int(Settings.kfoldSplits))
            for train, test in kfold.split(self.inputs, self.labels):
                Settings.logger.debug('train -  {}   |   test -  {}'.format(
                np.bincount(self.labels[train]), np.bincount(self.labels[test])))
            split = kfold.split(self.inputs, self.labels)
        else:
            Settings.logger.debug('Cross-domain manual CV')
            kfold = LeaveOneGroupOut()
            # Show distribution of cross split
            for train, test in kfold.split(self.inputs, self.labels, groups=self.groups):
                Settings.logger.debug('train -  {}   |   test -  {}'.format(
                    np.bincount(self.labels[train]), np.bincount(self.labels[test])))
            split = kfold.split(self.inputs, self.labels, groups=self.groups)
        # cycle split and fit, evaluate each time
        i = 1
        self.accuracy = []
        self.fscore = []
        self.precision = []
        self.recall = []
        for train_index, test_index in split:
            X_train = self.inputs[train_index]
            X_test = self.inputs[test_index]
            y_train = self.labels[train_index]
            y_test = self.labels[test_index]
            #model = self.network()

            # Train the model
            # model.fit(X_train, y_train, epochs=int(Settings.epoch), batch_size=5)  # if simple sequential model
            self.buildNetwork()
            fitResults = self.classifier.fit(X_train, y_train, class_weight=self.weights)
            evaluationResults = self.classifier.predict(X_test)
            #print(f"Accuracy for the fold no. {i} on the test set: {accuracy_score(y_test,(self.classifier.predict(X_test) > 0.5).astype('int32'))}")
            #print("accuracy as seen from model.evaluate: " + str(model.evaluate(X_test, y_test)[1]))  # if plain simple keras sequential model
            #print("accuracy as seen from model.evaluate: " + str(self.classifier.score(X_test, y_test)))          # if KerasClassifier wrapper
            m = Accuracy()
            m.update_state(y_test, evaluationResults)
            self.accuracy.append(float(m.result()))
            m = Precision()
            m.update_state(y_test, evaluationResults)
            self.precision.append(float(m.result()))
            m = Recall()
            m.update_state(y_test, evaluationResults)
            self.recall.append(float(m.result()))
            self.fscore.append(2 * (self.precision[-1] * self.recall[-1]) / (self.precision[-1] + self.recall[-1]))
            i += 1
            self.network = None
            self.classifier = None
        # destroy trained model to avoid interfering with other CV or prediction
        self.network = None
        self.classifier = None
        self.precision = {'mean': np.array(self.precision).mean(), 'std': np.array(self.precision).std()}
        self.accuracy = {'mean': np.array(self.accuracy).mean(), 'std': np.array(self.accuracy).std()}
        self.recall = {'mean': np.array(self.recall).mean(), 'std': np.array(self.recall).std()}
        self.fscore = {'mean': np.array(self.fscore).mean(), 'std': np.array(self.fscore).std()}
        Settings.logger.debug('MANUAL CV performances: ')
        Settings.logger.debug(
            "Accuracy: %0.2f (+/- %0.2f)" % (self.accuracy['mean'], self.accuracy['std'] * 2))
        Settings.logger.debug(
            "Recall: %0.2f (+/- %0.2f)" % (self.recall['mean'], self.recall['std'] * 2))
        Settings.logger.debug("Precision: %0.2f (+/- %0.2f)" % (
        self.precision['mean'], self.precision['std'] * 2))
        Settings.logger.debug(
            "F1: %0.2f (+/- %0.2f)" % (self.fscore['mean'], self.fscore['std'] * 2))


    def predict(self):
        self.output = {}
        Settings.logger.debug('Started prediction...')
        if not Settings.CrossDomain:  # use stratifiedKFold
            # get a new classifier since other one has been used for
            Settings.logger.debug('Started In-Domain training...')
            self.buildNetwork()
            self.classifier.fit(self.inputs, self.labels, class_weight=self.weights)
            Settings.logger.debug('Started In-Domain prediction...')
            for domain in self.parser.test:
                self.output[domain] = []
                for pair in self.parser.test[domain]:
                    fromConcept = Model.dataset[Model.dataset.index(pair[0])]
                    toConcept = Model.dataset[Model.dataset.index(pair[1])]
                    result = [fromConcept.title, toConcept.title, (self.classifier.predict(
                        np.array([self.getFeatures(fromConcept, toConcept, 'none')])) > 0.5).astype('int32')[0][0]]
                    self.output[domain].append(result)
            self.network = None
            self.classifier = None
        else:
            # cycle a domain
            for domain in self.parser.test:
                # build dataset with all domains except the one we wanna predict
                featuresSet = []
                labelSet = []
                for i in range(len(self.inputs)):
                    if self.groups[i] != domain:
                        featuresSet.append(self.inputs[i])
                        labelSet.append(self.labels[i])
                Settings.logger.debug('Started cross-domain training for domain ' + domain)
                Settings.logger.debug('Taining on ' + str(len(featuresSet)) + ' samples')
                self.buildNetwork()
                self.classifier.fit(np.array(featuresSet), np.array(labelSet), class_weight=self.weights)
                Settings.logger.debug('Started cross-domain prediction...')
                self.output[domain] = []
                # predict all pairs in this domain
                for pair in self.parser.test[domain]:
                    fromConcept = Model.dataset[Model.dataset.index(pair[0])]
                    toConcept = Model.dataset[Model.dataset.index(pair[1])]
                    self.output[domain].append([
                        fromConcept.title,
                        toConcept.title,
                        (self.classifier.predict(
                            np.array([self.getFeatures(fromConcept, toConcept, '')])) > 0.5).astype('int32')[0][0]
                    ])
                self.network = None
                self.classifier = None
        Settings.logger.debug('Found ' + str(sum([sum([pair[2] for pair in self.output[domain]]) for domain in self.output])) +
                              ' prereqs in a ' + str(
            sum([len([pair[2] for pair in self.output[domain]]) for domain in self.output])) + ' long testSet')
        # destroy trained model to avoid interfering with other CV or prediction
        self.network = None
        self.classifier = None

    def process(self, notifyProgress=None):

        # def notifyProgress(stepName, stepProgress = None)
        self.calculateFeatures()
        self.encodeInputOutputs()
        if Settings.crossValidateCV:
            notifyProgress(stepName='Auto CrossValidation')
            self.autoCV()
        if Settings.manualCV:
            notifyProgress(stepName='Manual CrossValidation')
            self.manualCV()
        if Settings.testsetPath is not None:
            notifyProgress(stepName='Prediction')
            self.predict()
        if Settings.manualCV or Settings.crossValidateCV:
            return {'accuracy': self.accuracy['mean'], 'recall': self.recall['mean'],
                     'precision': self.precision['mean'], 'f1': self.fscore['mean'], 'result': self.output}
        else:
            return {'result': self.output}

    # serializes Model.dataset, desired GraphMatrix and pairFeatures to build an array of inputs/labels pairs for neural net
    def classifierFormatter(self):    # resample = True changes results: why? shouldn't wheights account for unbalanced classes?!?
        Settings.logger.debug('Beginning dataset formatting')
        # init to dictionary of empty arrays wiyh domains as keys

        prereqData = {domain: [] for domain in Model.desiredGraph.domains}
        notPrereqData = {domain: [] for domain in Model.desiredGraph.domains}
        prereqLabel = {domain: [] for domain in Model.desiredGraph.domains}
        notPrereqLabel = {domain: [] for domain in Model.desiredGraph.domains}
        total = 0
        classRatio = {}
        for prereq in Model.desiredGraph.getPrereqs():
            for postreq in Model.desiredGraph.getPostreqs(prereq):
                for domain in Model.desiredGraph.getDomains(prereq, postreq):
                    label = Model.desiredGraph.getPrereq(prereq, postreq, domain)
                    # Only consider known relations since % of unknown is > 90% and biases the system to always output "UNKNOWN"
                    if label != Model.desiredGraph.unknown:
                        total = total+1
                        # counts every class occurrencies, creates new class if it hasn't yet encountered it
                        if not classRatio.__contains__(label):
                            classRatio[label] = 0
                        classRatio[label] += 1 # increase this class counter
                        prereqConcept = Model.dataset[Model.dataset.index(prereq)]
                        postreqConcept = Model.dataset[Model.dataset.index(postreq)]
                        feat = self.getFeatures(prereqConcept, postreqConcept, domain)
                        # feat = [random.choice([0, 1])]  # only one, random features: should return performance = 50%
                        # feat = [int(Model.desiredGraph.getPrereq(conceptA, conceptB))]   # truth oracle, should return performance = 100%

                        if label == Model.desiredGraph.isPrereq:
                            prereqData[domain].append(feat)
                            prereqLabel[domain].append(label)

                        if label == Model.desiredGraph.notPrereq:
                            notPrereqData[domain].append(feat)
                            notPrereqLabel[domain].append(label)
        totalNotPrereq = sum([len(notPrereqLabel[key]) for key in notPrereqLabel])
        totalPrereq = sum([len(prereqLabel[key]) for key in prereqLabel])
        if totalNotPrereq + totalPrereq != total:
            raise Exception("Not all labels are of prerequisition")
        if abs(classRatio[0] - classRatio[1]) != abs(totalNotPrereq - totalPrereq):
            raise Exception("Something wrong in classes count")
        # classRatio has same value as GraphMatrix.getStatistics()
        minorData = notPrereqData if totalNotPrereq - totalPrereq < 0 else prereqData
        biggerData = prereqData if totalNotPrereq - totalPrereq < 0 else notPrereqData

        minorLabel = notPrereqLabel if totalNotPrereq - totalPrereq < 0 else prereqLabel
        biggerLabel = prereqLabel if totalNotPrereq - totalPrereq < 0 else notPrereqLabel
        pickedIndex = []
        if Settings.resampleSmallerClass:
            # while we have differencies in number of classes...
            while abs(sum([len(notPrereqLabel[key]) for key in notPrereqLabel]) - sum([len(prereqLabel[key]) for key in prereqLabel])) > 0:
                # ...randomly duplicate smaller class samples
                randomDomain = random.choice([*minorLabel.keys()])
                randomIndex = random.choice(range(len(minorLabel[randomDomain])))
                pickedIndex.append(randomDomain + '-' + str(randomIndex))
                minorLabel[randomDomain].append(minorLabel[randomDomain][randomIndex])
                minorData[randomDomain].append(minorData[randomDomain][randomIndex])
            Settings.logger.debug("resampled a total of " + str(len(pickedIndex)) + " concepts")
        else:
            while abs(sum([len(notPrereqLabel[key]) for key in notPrereqLabel]) - sum([len(prereqLabel[key]) for key in prereqLabel])) > 0:
                # ...or randomly drop bigger class samples
                randomDomain = random.choice([*biggerLabel.keys()])
                randomIndex = random.choice(range(len(biggerLabel[randomDomain])))
                pickedIndex.append(randomDomain + '-' + str(randomIndex))
                biggerLabel[randomDomain].pop(randomIndex)
                biggerData[randomDomain].pop(randomIndex)
                # control we are have emptied random picked domain, if so remove that domain
                if len(biggerLabel[randomDomain]) == 0:
                    biggerLabel.pop(randomDomain)
                    biggerData.pop(randomDomain)
            Settings.logger.debug("dropped a total of " + str(len(pickedIndex)) + " concepts")

        if len(pickedIndex) != abs(classRatio[0] - classRatio[1]):
            raise Exception("Something wrong rebalancing: rebalanced " + str(
                len(pickedIndex)) + ", original difference " + str(abs(classRatio[0] - classRatio[1])))

        number_of_classes = 2 # 2 if classes are isPrereq/notPrereq, 3 if Unknown is allowed

        # different examples for class balancing through weights
        # if notPrereq is 1/10 of prereq samples: notPrereq errors should account ten times those of prereq
        #weights = {0: 1, 1: classRatio[0] / classRatio[1]}  # ratio between classes
        #weights = {0: classRatio[1], 1: classRatio[0]}      # opposite ratio: in the end ratios are the same as above
        # inverse ratio: in the end ratios are the same as above
        weights = {Model.desiredGraph.notPrereq: sum([len(prereqLabel[key]) for key in prereqLabel]),
                   Model.desiredGraph.isPrereq: sum([len(notPrereqLabel[key]) for key in notPrereqLabel])}
        #weights = {0: 1, 1: 1}  # should behave as if no weights were specified

        features = []
        labels = []
        groups = []
        for domain in prereqData.keys():
            features = features + [*prereqData[domain]]
            labels = labels + [*prereqLabel[domain]]
            groups = groups + [*([domain] * (len(prereqLabel[domain])))]

        for domain in notPrereqLabel.keys():
            features = features + [*notPrereqData[domain]]
            labels = labels + [*notPrereqLabel[domain]]
            groups = groups + [*([domain] * (len(notPrereqLabel[domain])))]

        Settings.logger.debug('Finished dataset formatting')
        self.weights = weights
        self.groups = groups
        self.inputSize = len(features[0])
        self.outputSize = number_of_classes
        return {'features': features, "desired": labels}

    def getFeatures(self, conceptA, conceptB, domain):
        features = []
        if Settings.useRefD:
            features.append(self.pairFeatures.getRefDistance(conceptA, conceptB))
            features.append(self.pairFeatures.getRefDistance(conceptB, conceptA))
            features.append(conceptA.features.totalIncomingLinks)
            features.append(conceptA.features.totalOutgoingLinks)
            features.append(conceptB.features.totalIncomingLinks)
            features.append(conceptB.features.totalOutgoingLinks)
        if Settings.useConceptLDA:
            features = features + conceptA.getFeatures().get_LDAVector()  # spread operator: *['a', 'b'] = a, b
        if Settings.useJaccard:
            features.append(self.pairFeatures.getJaccardSim(conceptA, conceptB))
            features.append(self.pairFeatures.getJaccardSim(conceptB, conceptA))
        if Settings.useContainsLink:
            features.append(self.pairFeatures.getLink(conceptA, conceptB))
            features.append(self.pairFeatures.getLink(conceptB, conceptA))
        if Settings.useLDACrossEntropy:
            features.append(self.pairFeatures.getLDACrossEntropy(conceptA, conceptB))
            features.append(self.pairFeatures.getLDACrossEntropy(conceptB, conceptA))
        if Settings.useLDA_KLDivergence:
            features.append(self.pairFeatures.getLDA_KLDivergence(conceptA, conceptB))
            features.append(self.pairFeatures.getLDA_KLDivergence(conceptB, conceptA))
        if Settings.contains:
            features.append(self.pairFeatures.getContainsTitle(conceptA, conceptB))
            features.append(self.pairFeatures.getContainsTitle(conceptB, conceptA))
        return features
