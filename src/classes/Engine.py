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
    """ Engine

    Focus point of the execution, this class instantiates all the requested
    features, format them and feeds it to the Neural Network. returns the four
    performance indicators along with the labels for the evaluated pairs

    ...

    Attributes
    ----------
    pairFeatures : PairFeatures
        container for pair features
    parser : Parser
        Parser object used to load input files
    inputSize : int
        tracks the dimension of the first layer of the network
    outputSize : int
        tracks the dimension of the output layer of the network (default: 2)
    groups : [string]
        one string for each sample submitted to the NN indicating teh sample's domain
    labels : [bool]
        one bool for each sample submitted to the NN indicating if it's prereq or not
    inputs : np.array
        contains the samples used to train the NN
    classifier : KerasClassifier
        Keras Classifier: a wrapper around the actual NN which allows for training and evaluation of samples
    network : function()
        contains the function to build the layers of the NN, used to initialize the classifier
    weights : {0: int, 1: int}
        Dictionary with a weight for not-prereq samples (key 0) and prereq ones (key 1)
        Allows for resampling of unbalanced dataset, helps avoid unbalanced datasets bias
    accuracy : int
        contains the accuracy performance metric
    recall : int
        contains the recall performance metric
    precision : int
        contains the precision performance metric
    fscore : int
        contains the fscore performance metric
    totalPrediction : int
        counter that traces how many classification were evaluated
    correctProbabilities : {}
        Keys: probability (range [0, 1]). Value: a counter that tracks how
        many correct labels were given with said confidence
    wrongProbabilities : {}
        Keys: probability (range [0, 1]). Value: a counter that tracks how
        many incorrect labels were given with said confidence

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    pairFeatures = None
    parser = Parser()

    inputSize = None
    outputSize = None
    groups = None
    labels = None
    inputs = None
    classifier = None
    network = None
    weights = None

    accuracy = None
    recall = None
    precision = None
    fscore = None

    totalPredictions = 0
    correctProbabilities = {}
    wrongProbabilities = {}

    def __init__(self):
        """Instantiates the PairFeatures and correct/wrong Probabilities

        If a PairFeatures dump is found in a folder specified by the Settings module
        it loads it. Otherwise an empty PairFeatrues object is created
        """
        if os.path.exists(Settings.pairFeaturesPickle):
            with open(Settings.pairFeaturesPickle, 'rb') as file:
                self.pairFeatures = pickle.load(file)
        else:
            self.pairFeatures = PairFeatures()
        for i in range(50, 101):    # initialize probabilities from 0 to 100% to zero as they are incremental
            self.correctProbabilities[str(0+i/100)] = 0
            self.wrongProbabilities[str(0+i/100)] = 0

    def calculateFeatures(self):
        """instantiates the *Extractor classes and uses them to calculate required features

        Uses FeatureExtractor to calculate text features and MetaExtractor to calculate
        structured features.
        Signle Concept features are saved directly in Concept.features while PairFeatures
        are saved in this object pairFeatures attribute
        """
        # text features
        feature = FeatureExtractor(self.pairFeatures)
        ## single concept
        Settings.logger.debug('Starting sencence extraction (might take a lot)...')
        start_time = time.time()
        feature.extractSentences()
        elapsed_time = time.time() - start_time
        Settings.logger.debug('Using Cache: ' + str(Settings.useCache and os.path.exists(Settings.conceptsPickle)) +
                              ", Annotation Elapsed time: " + str(elapsed_time))
        feature.extractNounsVerbs()
        feature.LDA()
        feature.containsTitle()

        ## pair concepts
        feature.jaccardSimilarity()
        feature.LDACrossEntropy()

        # meta features
        Settings.logger.info("Fetching Meta Info...")
        meta = MetaExtractor(self.pairFeatures)
        ## single concept
        meta.annotateConcepts()
        ## pair concepts
        meta.extractLinkConnections()
        meta.referenceDistance()

    def encodeInputOutputs(self):
        """Creates the numpy arrays for input samples and their labels (required by Keras)

        """
        encoder = LabelEncoder()
        result_set = self.classifierFormatter()
        x = np.array(result_set['features'])
        encoder.fit(result_set['desired'])

        # from generic label to integer: ['a', 'a', 'b', 1, 1, 1, 1]->[1, 1, 2, 0, 0, 0, 0]
        encoded_y = encoder.transform(result_set['desired'])
        self.labels = encoded_y
        self.inputs = x

    def buildNetwork(self):
        """Utility method that creates the NN and instantiates the Classifier

        The network is created using the sequential model From Keras.
        This method gets called after the formatting of inputs and outputs has been done
        and relies on the inputSize and outputSize value to correctly build the first
        and last layers of the net. The NN and Classifier are then stored in this class
        corresponding attributes
        """
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
        """Uses the cross-validate method from SciKit to calculate Cross-Validation performance of the model

        It creates the classifier, creates the Folds for applying KFold distinguishing between
        cross-domain or in-domain scenario, logs distribution of the dataset and perform the CV.
        Calculates the average of each metrics and stores it in the associated attribute
        """
        self.buildNetwork()
        scoring = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_micro',
                   'average_precision', 'precision_micro', 'recall_micro']
        # train on all domains->use stratifiedKFold
        if not Settings.CrossDomain:
            Settings.logger.debug('In-domain cross-validation')
            kfold = StratifiedShuffleSplit(n_splits=int(Settings.kfoldSplits), test_size=1/int(Settings.kfoldSplits))
            # Show to console the distribution of cross split for debug
            for train, test in kfold.split(self.inputs, self.labels):
                Settings.logger.debug('train -  {}   |   test -  {}'.format(
                np.bincount(self.labels[train]), np.bincount(self.labels[test])))
            # Actually perform CV
            scores = cross_validate(self.classifier, self.inputs, self.labels, n_jobs=-1, scoring=scoring, cv=kfold,
                                    fit_params={'class_weight': self.weights})
        # Use LeaveOneGroupOut KFold
        else:
            Settings.logger.debug('Cross-domain cross-validation')
            kfold = LeaveOneGroupOut()
            # Show distribution of cross split
            for train, test in kfold.split(self.inputs, self.labels, groups=self.groups):
                Settings.logger.debug('train -  {}   |   test -  {}'.format(
                np.bincount(self.labels[train]), np.bincount(self.labels[test])))
            # Actually perform CV
            scores = cross_validate(self.classifier, self.inputs, self.labels, n_jobs=-1, scoring=scoring, cv=kfold,
                                    groups=self.groups, fit_params={'class_weight': self.weights})

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
        """Manually calculates Cross-Validation performance of the model

        It creates the classifier, creates the Folds for applying KFold distinguishing between
        cross-domain or in-domain scenario, logs distribution of the dataset.
        CV is done cycling through each KFold (i.e. set of indexes over the dataset) manually
        to gain insight on the performance and extract confidence of the labels extracted.
        At the end of each Fold cycle the model is reset to avoid carrying any information from previous Fold.
        Finally calculates the average of each metrics and stores it in the associated attribute.
        """
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
            test_domains = np.array(self.groups)[test_index]
            # Train the model
            self.buildNetwork()
            fitResults = self.classifier.fit(X_train, y_train, class_weight=self.weights)
            evaluationResults = self.classifier.predict(X_test)
            evaluationProbabilities = self.classifier.predict_proba(X_test)
            for i in range(len(evaluationResults)-1):
                if y_test[i] == evaluationResults[i][0]: # i.e. prediction is correct
                    prob = round(evaluationProbabilities[i][y_test[i]], 2)
                    self.correctProbabilities[str(prob)] = self.correctProbabilities[str(prob)] + 1
                else:
                    prob = round(evaluationProbabilities[i][int(not y_test[i])], 2)
                    self.wrongProbabilities[str(prob)] = self.wrongProbabilities[str(prob)] + 1
            self.totalPredictions = self.totalPredictions + len(evaluationResults)

            # Calculate performances for this CV run
            m = Accuracy()
            m.update_state(y_test, evaluationResults)
            self.accuracy.append(float(m.result()))
            m = Precision()
            m.update_state(y_test, evaluationResults)
            self.precision.append(float(m.result()))
            m = Recall()
            m.update_state(y_test, evaluationResults)
            self.recall.append(float(m.result()))
            if ((self.precision[-1] + self.recall[-1]) != 0):
                self.fscore.append(2 * (self.precision[-1] * self.recall[-1]) / (self.precision[-1] + self.recall[-1]))
            else:
                self.fscore.append(0)
            i += 1
            self.network = None
            self.classifier = None

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
        """Labels new, unseen Concept pairs with a Classifier trained on the whole Dataset

        CrossDomain Scenario:
        Creates a Classifier and trains it on the whole dataset (already formatted, stored in
        self.inputs and self.labels). Then uses it to labels all Concept pairs in self.parser.test

        InDomain Scenario:
        for each domain in self.parser.test creates a Classifier and trains it on all dataset domains
        but its own.
        It then uses the classifier to label all pairs in self.parser.test of the subject domain

        In both Scenarios result are grouped by domain in the output dictionary.
        Thus self.output will have the following structure:
        {domain_1: [bool], domain_2: [bool], domain_3: [bool], ...}
        """
        self.output = {}
        Settings.logger.debug('Started prediction...')
        if not Settings.CrossDomain:  # use stratifiedKFold
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

    def process(self, widget=None):
        """Entrypoint: this public method manages the calculation and formatting of features,
           runs the CV and finds labels for new pairs

        Args:
            widget: Callback to notify progress of execution, defaults to None.
                can be used e.g. to notify the UI of current execution point

        Returns:
            A dictionary containing:
                - the performance of the model, each metric addressed by its name (only if CV was performed)
                - Confidence arrays for correct and wrong labels during CV (only if CV was performed)
                - the labels for new pairs, under the key "results"
        """
        if widget == None:
            def temp(stepName, stepProgress = None):
                pass
            widget = temp
        widget.emit('Calculating features...')
        Settings.logger.debug('after Calculating Features')
        self.calculateFeatures()
        widget.emit('Encoding dataset...')
        Settings.logger.debug('after Encoding dataset')
        self.encodeInputOutputs()
        if Settings.crossValidateCV:
            widget.emit('Cross Validating...')
            self.autoCV()
        if Settings.manualCV:
            widget.emit('Cross Validating...')
            self.manualCV()
        Settings.logger.debug('after Cross Validating')
        if Settings.testsetPath is not None:
            widget.emit('Evaluating Requested pairs...')
            Settings.logger.debug('after Evaluating Requested pairs')
            self.predict()
        if Settings.manualCV or Settings.crossValidateCV:
            return {'accuracy': self.accuracy['mean'], 'recall': self.recall['mean'],
                     'precision': self.precision['mean'], 'f1': self.fscore['mean'], 'result': self.output,
                    'correctProbabilities': self.correctProbabilities, 'wrongProbabilities': self.wrongProbabilities}
        else:
            return {'result': self.output}

    def classifierFormatter(self):
        """ Serializes the inputs (aka "features") and the output labels (aka "desired") needed by the NN

        It cycles through all concept pairs saved in Model.desiredGraphMatrix.
            Since a pair can be present for multiple domains it cycles on domains.
                Gets all features for the given Concepts
                Depending on the direction of the prerequisite it appends them to the "prereqData"
                or "notPrereqData" dictionary (under the corresponding domain),
                Appends the Model.desiredGraphMatrix of the current Concept pair to the "desired" array.
        Saves locally some info on the formatted dataset such as size of features vectors
        It looks how unbalanced the dataset is and calculates weights and saves them
        Finally merges both "prereq*" and "notPrereq*" arrays and returns them.

        Returns:
            A dict containing two arrays:
            {'features': [], 'desired': []}
        """
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
                    if label != Model.desiredGraph.unknown:
                        total = total+1
                        # counts every class occurrencies, creates new class if it hasn't yet encountered it
                        if not classRatio.__contains__(label):
                            classRatio[label] = 0
                        classRatio[label] += 1 # increase this class counter
                        prereqConcept = Model.dataset[Model.dataset.index(prereq)]
                        postreqConcept = Model.dataset[Model.dataset.index(postreq)]
                        feat = self.getFeatures(prereqConcept, postreqConcept, domain)

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

        number_of_classes = 2
        # inverse ratio: if notPrereq is 1/10 of prereq samples: notPrereq errors should account ten times those of prereq
        weights = {Model.desiredGraph.notPrereq: sum([len(prereqLabel[key]) for key in prereqLabel]),
                   Model.desiredGraph.isPrereq: sum([len(notPrereqLabel[key]) for key in notPrereqLabel])}
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
        """Serializes all features of a given pair of concepts

        Builds a single array by merging:
            - all single Concept features of Concept A (from Model.dataset)
            - all single Concept features of Concept B (from Model.dataset)
            - all pair Features of the pair A, B (from self.pairFeatures)
            - the domain of the pair

        Returns:
             A single array of integers, each being a feature
        """
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
