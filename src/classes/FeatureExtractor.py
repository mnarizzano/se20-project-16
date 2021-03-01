__author__ = "Moggio Alessio, Parizzi Andrea"
__license__ = "Public Domain"
__version__ = "1.0"

from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.models import Sequential
from ufal.udpipe import Model, Pipeline, ProcessingError
from Parser import Parser
from Model import Model as MyModel
from conllu import parse
from Settings import Settings
from sklearn.feature_extraction.text import CountVectorizer
import time
import pandas as pd
from gensim import matutils, models
import scipy.sparse
from math import log2
from gensim.models import Word2Vec
import pickle

class FeatureExtractor:
    """Utility class used to extract Text features.


    Each feature method first checks if this isn't already calculated, in which case skips
    and before returning caches the pairFeatures to disk if they got modified
    """
    pairFeatures = None
    parser = Parser()

    def __init__(self, pairFeatures):
        """Constructor

        Args:
            pairFeatures: reference to the singleton instance of PairFeatures holding
                all features related to couples of concepts

        """
        self.pairFeatures = pairFeatures

    def cache(self):
        """Saves the current pairFeatures object to disk
        """
        Settings.logger.debug('Caching pairFeatures...')
        pickle.dump(self.pairFeatures, open(Settings.pairFeaturesPickle, "wb+"))

    def extractSentences(self):
        """For each Concept in the dataset uses a udpipeModel to extract sentences from their text
        """
        loaded = (MyModel.dataset[len(MyModel.dataset)-1].features.annotatedSentences is not None) and \
                 MyModel.dataset[len(MyModel.dataset) - 1].features.annotatedSentences != []
        if not loaded: # if already present it has been loaded from pickle
            udpipeModel = Model.load(Settings.udpipeModelPath)
            if udpipeModel is None:
                raise Exception("Couldn't find UdPipe model: " + Settings.udpipeModelPath)
            pipeline = Pipeline(udpipeModel, 'tokenize',
                                Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
            error = ProcessingError()

            for concept in MyModel.dataset:
                concept.features.conllu = pipeline.process(concept.content, error)
                concept.features.annotatedSentences = parse(concept.features.conllu)  # This are annotated sentences
                for sentence in concept.features.annotatedSentences:
                    concept.features.sentences.append(sentence.metadata['text'])
            self.parser.cache()

    def LDA(self):
        """Calculates LDA for each Concept using a GenSim model
        """
        if (not hasattr(MyModel.dataset[-1].features, 'ldaVector')) or\
                (MyModel.dataset[-1].features.ldaVector == []):
            Settings.logger.debug('Starting LDA Calculation with ' + str(Settings.numberOfTopics) + ' topics')
            start_time = time.time()
            vectorizer = CountVectorizer()
            corpus = [[noun['lemma']for noun in concept.features.nounsList]for concept in MyModel.dataset]
            data = vectorizer.fit_transform([' '.join(concept) for concept in corpus])
            self.dtm = pd.DataFrame(data.toarray(), columns=vectorizer.get_feature_names())
            self.dtm.index = [concept.title for concept in MyModel.dataset]
            tdm = self.dtm.transpose()
            sparse_count = scipy.sparse.csr_matrix(tdm)
            corpus = matutils.Sparse2Corpus(sparse_count)
            id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())
            lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=Settings.numberOfTopics, passes=15)
            corpus_transformed = lda[corpus]
            ldaVectors = dict(list(zip(self.dtm.index, [a for a in corpus_transformed])))
            for concept in MyModel.dataset:
                if sum([p[1] for p in ldaVectors[concept.title]]) < 0.1:
                    Settings.logger.error("Error something wrong in topic modeling: Concept '" +
                                          concept.title + "' is not assigned to any topic")
                leftOutProbability = (1-sum([p[1] for p in ldaVectors[concept.title]]))/\
                                     (Settings.numberOfTopics-len(ldaVectors[concept.title]))
                concept.features.ldaVector = [leftOutProbability]*Settings.numberOfTopics
                for ldaComponent in ldaVectors[concept.title]:
                    concept.features.ldaVector[ldaComponent[0]] = ldaComponent[1]
            elapsed_time = time.time() - start_time
            Settings.logger.debug('LDA calculation Elapsed time: ' + str(elapsed_time))
            self.parser.cache()
        else:
            Settings.logger.debug('Skipped LDA calculation')

    def kl_divergence(self, p, q):
        """Calculate the kl divergence KL(P || Q)
        """
        return sum(p[i] * log2(p[i] / q[i]) for i in range(len(p)))

    def entropy(self, p):
        """Calculate entropy H(P)
        """
        return -sum([p[i] * log2(p[i]) for i in range(len(p))])

    def cross_entropy(self, p, q):
        """Calculate cross entropy H(P, Q)
        """
        return self.entropy(p) + self.kl_divergence(p, q)

    def LDACrossEntropy(self):
        """Calculate cross entropy of the LDA vectors between 2 Concepts
        """
        Settings.logger.debug('Calculating LDA cross-entropy...')
        if not self.pairFeatures.LDACrossEntropyLoaded():
            for conceptA in MyModel.dataset:
                conceptA.features.LDAEntropy = self.entropy(conceptA.features.ldaVector)    # this annotates ALL concepts
                for conceptB in MyModel.dataset:
                    self.pairFeatures.setLDACrossEntropy(conceptA, conceptB,
                        self.cross_entropy(conceptA.features.ldaVector, conceptB.features.ldaVector))
                    self.pairFeatures.setLDA_KLDivergence(conceptA, conceptB,
                        self.kl_divergence(conceptA.features.ldaVector, conceptB.features.ldaVector))
            self.parser.cache()
        else:
            Settings.logger.debug('Skipping LDA since it was already present')

    def sentenceOfFocus(self, concept, annotatedSentence):
        """Checks if any sentence of a Concept Text includes the title of another Concept
        """
        # check if in sentence appears title of concept
        contains = concept.title.lower() in annotatedSentence.metadata['text'].lower()

        # check if in lemmatized sentence appears title of concept
        lemmatized = False
        for token in annotatedSentence:
            lemmatized = (token['lemma'].lower() == concept.title.lower) or lemmatized

        if (contains or lemmatized):
            Settings.logger.debug("Found concept '" + concept.title + "' in sentence '" + annotatedSentence.metadata['text'] + "'")

        return (contains or lemmatized)

    def extractNounsVerbs(self):
        """Filter the annotation of a Concept Text separating Nouns from Verbs, returns a set for each.
        """
        loaded = (MyModel.dataset[len(MyModel.dataset) - 1].features.nounsList is not None) and \
                 (MyModel.dataset[len(MyModel.dataset) - 1].features.nounsList != []) and \
                 (MyModel.dataset[len(MyModel.dataset) - 1].features.verbsSet is not None) and \
                 (MyModel.dataset[len(MyModel.dataset) - 1].features.verbsSet) != set() and \
                 (MyModel.dataset[len(MyModel.dataset) - 1].features.nounsPlain is not None) and \
                 (MyModel.dataset[len(MyModel.dataset) - 1].features.nounsPlain != [])
        if not loaded:
            for concept in MyModel.dataset:
                concept.features.nounsList = []
                concept.features.nounsPlain = []
                concept.features.nounsSet = set()

                concept.features.verbsList = []
                concept.features.verbsPlain = []
                concept.features.verbsSet = set()
                for annotatedSentence in concept.features.annotatedSentences:
                    for token in annotatedSentence:
                        if token['upos'] == 'NOUN':
                            concept.features.nounsList.append(token)
                            concept.features.nounsPlain.append(token['lemma'])
                            concept.features.nounsSet.add(token['lemma'])
                        if token['upos'] == 'VERB':
                            concept.features.verbsList.append(token)
                            concept.features.verbsPlain.append(token['lemma'])
                            concept.features.verbsSet.add(token['lemma'])

    def nounsVerbs2Set(self):
        """Transforms and array into a Set
        """
        for concept in MyModel.dataset:
            concept.features.nounsSet = set(f['lemma'] for f in concept.features.nouns)

    def jaccardSimilarity(self):
        """Calculates the jaccardSimilarity between text from a pair of Concepts
        """
        if not self.pairFeatures.jaccardLoaded():
            Settings.logger.debug('Calculating Jaccard Similarity')
            self.extractNounsVerbs()
            for conceptA in MyModel.dataset:
                for conceptB in MyModel.dataset:
                    if conceptB.domain == conceptA.domain:
                        if len(conceptA.features.nounsSet.union(conceptB.features.nounsSet))==0:
                            js = 0
                        else:
                            js = len(conceptA.features.nounsSet.intersection(conceptB.features.nounsSet))/\
                                 len(conceptA.features.nounsSet.union(conceptB.features.nounsSet))
                        self.pairFeatures.setJaccardSimilarity(conceptA, conceptB, js)
            self.cache()
        else:
            Settings.logger.debug('Skipping jaccard cause it was cached')

    def trainLSTMNet(self, inputs, outputs):
        """PoC: private method, testing LSMT Networks using Glove model from isti.cnr.
        """
        model = self.loadWordEmbeddings()
        def build_lstm():
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(10, 200), return_sequences=False))
            model.add(Dense(1), activation='sigmoid')
            model.compile(optimizer='adam', loss='binary_crossentropy')

        # return trained lstm for this round
        return build_lstm

    def containsTitle(self):
        """Checks if any sentence of a Concept Text includes the title of another Concept
        """
        if not self.pairFeatures.containsTitleLoaded():
            Settings.logger.debug('Calculating Contains Title')
            for conceptA in MyModel.dataset:
                for conceptB in MyModel.dataset:
                    self.pairFeatures.setContainsTitle(conceptA, conceptB, conceptB.content.count(conceptA.title))
            self.cache()
        else:
            Settings.logger.debug('Skipping contains title cause it was cached')

    def loadWordEmbeddings(self):
        """Loads Glove model from isti.cnr.
        """
        model = Word2Vec.load(Settings.glove_WIKI)  # glove model
        Settings.logger.info('Loaded Glove model')
        return model

if __name__ == '__main__':
    from PairFeatures import PairFeatures
    f = FeatureExtractor(PairFeatures())
    f.loadWordEmbeddings()