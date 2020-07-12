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
        print(list(zip([a for a in corpus_transformed], self.dtm.index)))
        print('stop')

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


    def NounsVerbs2Set(self):   # This is batch List2Set
        for concept in MyModel.dataset:
            concept.features.nounsSet = set(f['lemma'] for f in concept.features.nouns)