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

class FeatureExtractor:

    pairFeatures = None

    def __init__(self, pairFeatures):
        self.pairFeatures = pairFeatures


    def extractSentences(self):
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
                '''
                if concept.title == 'Distribuzione di probabilità a priori':		# wiki page for test
                    Settings.logger.debug("Concept CONLLU: '" + concept.features.conllu + "'")
                    Settings.logger.debug("Parsed CONLLU: '" + str(concept.features.get_numberOfSentences()) + "'")
                    for sentence in concept.features.annotatedSentences:
                        self.sentenceOfFocus(MyModel.dataset[MyModel.dataset.index('Probabilità condizionata')], sentence)
                '''


    def LDA(self):
        numberOfTopics = 10  # TODO: either read from Settings or Configuration(and have Engine write to it->easier for configuration), or pass to method from engine
        if (not hasattr(MyModel.dataset[-1].features, 'ldaVector')) or (MyModel.dataset[-1].features.ldaVector == []):
            Settings.logger.debug('Starting LDA Calculation with ' + str(numberOfTopics) + ' topics')
            start_time = time.time()
            vectorizer = CountVectorizer()
            corpus = [[noun['lemma']for noun in concept.features.nounsList]for concept in MyModel.dataset] # TODO already calculated in nounsPlain
            data = vectorizer.fit_transform([' '.join(concept) for concept in corpus]) # TODO: play with stop-words: Ita e/o TF-ID (parole comuni a tutti i concetti non portano informazione per quel singolo concetto)
            # TODO: move dtm to another location
            self.dtm = pd.DataFrame(data.toarray(), columns=vectorizer.get_feature_names())
            # print(dtm.shape) # is #of concepts x #of different words in corpus
            self.dtm.index = [concept.title for concept in MyModel.dataset]
            # print(self.dtm)
            tdm = self.dtm.transpose()
            sparse_count = scipy.sparse.csr_matrix(tdm)
            corpus = matutils.Sparse2Corpus(sparse_count)
            id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())
            lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=numberOfTopics, passes=15)  # TODO: play with these numbers
            corpus_transformed = lda[corpus]
            ldaVectors = dict(list(zip(self.dtm.index, [a for a in corpus_transformed])))
            # ldaVectors = {'concept.title':[topicNumber, probability of concept belonging to topicNumber]}
            # topicNumber is an int in range [0, numberofTopics)
            # probability is a float in range [0, 1]
            for concept in MyModel.dataset: # Note: dataset and ldaVectors have same order, might speed up going by index instead of dictionary(dictionary is safer)
                if sum([p[1] for p in ldaVectors[concept.title]]) < 0.1:
                    Settings.logger.error("Error something wrong in topic modeling: Concept '" +  concept.title + "' is not assigned to any topic")
                leftOutProbability = (1-sum([p[1] for p in ldaVectors[concept.title]]))/(numberOfTopics-len(ldaVectors[concept.title]))
                concept.features.ldaVector = [leftOutProbability]*numberOfTopics    # TODO: qwertyuiop WARNING pseudo-randomly assigned LDA confidence for concepts not explicitly in LDA output
                for ldaComponent in ldaVectors[concept.title]:
                    concept.features.ldaVector[ldaComponent[0]] = ldaComponent[1]
            elapsed_time = time.time() - start_time
            Settings.logger.debug('LDA calculation Elapsed time: ' + str(elapsed_time))
        else:
            Settings.logger.debug('Skipped LDA calculation')

    # calculate the kl divergence KL(P || Q)
    def kl_divergence(self, p, q):
        return sum(p[i] * log2(p[i] / q[i]) for i in range(len(p)))

    # calculate entropy H(P)
    def entropy(self, p):
        return -sum([p[i] * log2(p[i]) for i in range(len(p))])

    # calculate cross entropy H(P, Q)
    def cross_entropy(self, p, q):
        return self.entropy(p) + self.kl_divergence(p, q)

    def LDACrossEntropy(self):
        for conceptA in MyModel.dataset:
            conceptA.features.LDAEntropy = self.entropy(conceptA.features.ldaVector)    # this annotates ALL concepts
            for conceptB in MyModel.dataset:
                self.pairFeatures.setLDACrossEntropy(conceptA, conceptB,
                    self.cross_entropy(conceptA.features.ldaVector, conceptB.features.ldaVector))
                self.pairFeatures.setLDA_KLDivergence(conceptA, conceptB,
                    self.kl_divergence(conceptA.features.ldaVector, conceptB.features.ldaVector))


    def sentenceOfFocus(self, concept, annotatedSentence):
        # check if in sentence appears title of concept
        contains = concept.title.lower() in annotatedSentence.metadata['text'].lower()

        # check if in lemmatized sentence appears title of concept
        lemmatized = False
        for token in annotatedSentence:
            lemmatized = (token['lemma'].lower() == concept.title.lower) or lemmatized

        if (contains or lemmatized):
            Settings.logger.debug("Found concept '" + concept.title + "' in sentence '" + annotatedSentence.metadata['text'] + "'")

        # TODO: full match may be too strict, maybe use partial match and return double instead of boolean
        # TODO: also check for synonims of concept as it might not be always written in the same way
        return (contains or lemmatized)


    def extractNounsVerbs(self):
        # TODO check all nouns/verbs/corpus * list/set combinations
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


    def nounsVerbs2Set(self):   # This is batch List2Set
        for concept in MyModel.dataset:
            concept.features.nounsSet = set(f['lemma'] for f in concept.features.nouns)

    # TODO: only calculate pairFeatures for same domain (to speed up execution) -> no more separation by domain
    def jaccardSimilarity(self):    # calculated over Nouns
        self.extractNounsVerbs()
        for conceptA in MyModel.dataset:
            for conceptB in MyModel.dataset:
                if conceptB.domain == conceptA.domain:  # To speed up, not possible if don't have domains
                    js = len(conceptA.features.nounsSet.intersection(conceptB.features.nounsSet))/\
                         len(conceptA.features.nounsSet.union(conceptB.features.nounsSet))
                    self.pairFeatures.setJaccardSimilarity(conceptA, conceptB, js)

    def referenceDistance(self, conceptA, conceptB):  # using EQUAL weights
        num1 = 0
        num2 = 0
        den1 = 0
        den2 = 0
        for concept in MyModel.dataset:
            num1 += (self.pairFeatures.features[concept.id][conceptB.id].link *
                        self.pairFeatures.features[conceptA.id][concept.id].link)
            num2 += (self.pairFeatures.features[concept.id][conceptA.id].link *
                        self.pairFeatures.features[conceptB.id][concept.id].link)
            den1 += (self.pairFeatures.features[conceptA.id][concept.id].link)
            den2 += (self.pairFeatures.features[conceptB.id][concept.id].link)
        
        if (den1 != 0 and den2 != 0):
            dist = (num1/den1) - (num2/den2)
            self.pairFeatures.setReferenceDistance(conceptA, conceptB, dist)
        else:
            # if den1 or den2 = 0, it means that A and B are no prerequisites
            self.pairFeatures.setReferenceDistance(conceptA, conceptB, 0)

    # TODO: instead of theese call this_Feature_Extractor_instance.pairFeatures.get...  (eventually .getPairFeatures().get..)
    def getRefDistance(self, conceptA, conceptB):
        return self.pairFeatures.getRefDistance(conceptA, conceptB)

    def getJaccardSim(self, conceptA, conceptB):
        return self.pairFeatures.getJaccardSim(conceptA, conceptB)

    def getLDACrossEntropy(self, conceptA, conceptB):
        return self.pairFeatures.getLDACrossEntropy(conceptA, conceptB)