from ufal.udpipe import Model, Pipeline, ProcessingError
from Parser import Parser
from Model import Model as MyModel
from conllu import parse
from Settings import Settings
from sklearn.feature_extraction.text import CountVectorizer
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
import pandas as pd
from gensim import matutils, models
import scipy.sparse


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
        common_dictionary = Dictionary(common_texts)
        common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
        # Train the model on the corpus.
        #lda = LdaModel(common_corpus, num_topics=10)

    def documentTermMatrix(self):
        vectorizer = CountVectorizer()
        corpus = [[noun['lemma']for noun in concept.features.nounsList]for concept in MyModel.dataset] # TODO already calculated in nounsPlain
        data = vectorizer.fit_transform([' '.join(concept) for concept in corpus]) # TODO: play with stop-words: Ita e/o TF-ID (parole comuni a tutti i concetti non portano informazione per quel singolo concetto)
        # TODO: move dtm to another location
        self.dtm = pd.DataFrame(data.toarray(), columns=vectorizer.get_feature_names())
        # print(dtm.shape) # is #of concepts x #of different words in corpus
        self.dtm.index = [concept.title for concept in MyModel.dataset]
        print(self.dtm)
        tdm = self.dtm.transpose()
        sparse_count = scipy.sparse.csr_matrix(tdm)
        corpus = matutils.Sparse2Corpus(sparse_count)
        id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())
        lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=10, passes=15)  # TODO: play with these numbers
        corpus_transformed = lda[corpus]
        # print(list(zip([a for a in corpus_transformed], self.dtm.index)))
        # [p for p in list(zip([a for a in corpus_transformed], self.dtm.index))] #cercare il concetto t.c. c.title == p[1] e fare c.features.lda = p[0] (in realtà sarebbe meglio fare un dizionario con chiave = indice del topic e valore = prob di appartenere a quel topic)
        # print('stop')

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

    # TODO: only calculate pairFeatures for same domain (to speed up execution)
    def jaccardSimilarity(self):
        self.extractNounsVerbs()
        for conceptA in MyModel.dataset:
            for conceptB in MyModel.dataset:
                if conceptB.domain == conceptA.domain:
                    # jaccardSimilarity is symmetric and is automatically added by PairFeatures to both A->B and B->A
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
        
        dist = (num1/den1) - (num2/den2)
        self.pairFeatures.setReferenceDistance(conceptA, conceptB, dist)
