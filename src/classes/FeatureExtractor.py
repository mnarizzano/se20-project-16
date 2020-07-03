from ufal.udpipe import Model, Pipeline, ProcessingError
from Parser import Parser
from Model import Model as MyModel
from conllu import parse
from Settings import Settings

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
        return (contains or lemmatized)


    def extractNounsVerbs(self):
        loaded = (MyModel.dataset[len(MyModel.dataset) - 1].features.nouns is not None) and \
                 MyModel.dataset[len(MyModel.dataset) - 1].features.nouns != []
        if not loaded:
            for concept in MyModel.dataset:
                concept.features.nounsList = []
                concept.features.verbsList = []
                concept.features.nounsSet = set()
                concept.features.verbsSet = set()
                for annotatedSentence in concept.features.annotatedSentences:
                    for token in annotatedSentence:
                        if token['upos'] == 'NOUN':
                            concept.features.nounsList.append(token)
                            concept.features.nounsSet.add(token['lemma'])
                        if token['upos'] == 'VERB':
                            concept.features.verbsList.append(token)
                            concept.features.verbsSet.add(token['lemma'])


    def NounsVerbs2Set(self):   # This is batch List2Set
        for concept in MyModel.dataset:
            concept.features.nounsSet = set(f['lemma'] for f in concept.features.nouns)

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
