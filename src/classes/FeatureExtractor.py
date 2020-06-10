from ufal.udpipe import Model, Pipeline, ProcessingError
from Parser import Parser
from Model import Model as MyModel
from conllu import parse
from Settings import Settings

class FeatureExtractor:

    udpipeModelPath = '../resources/Model/italian-isdt-ud-2.5-191206.udpipe'    # TODO move to Settings

    def extractSentences(self):
        udpipeModel = Model.load(self.udpipeModelPath)
        pipeline = Pipeline(udpipeModel, 'tokenize',
                            Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        error = ProcessingError()

        for concept in MyModel.dataset:
            concept.features.conllu = pipeline.process(concept.content, error)
            concept.features.annotatedSentences = parse(concept.features.conllu)  # This are annotated sentences
            for sentence in concept.features.annotatedSentences:
                concept.features.sentences.append(sentence.metadata['text'])
            if concept.title == 'Distribuzione di probabilità a priori':		# wiki page for test
                Settings.logger.debug("Concept CONLLU: '" + concept.features.conllu + "'")
                Settings.logger.debug("Parsed CONLLU: '" + str(concept.features.get_numberOfSentences()) + "'")
                for sentence in concept.features.annotatedSentences:
                    self.sentenceOfFocus(MyModel.dataset[MyModel.dataset.index('Probabilità condizionata')], sentence)


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
