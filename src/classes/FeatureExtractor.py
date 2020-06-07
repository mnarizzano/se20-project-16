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
            if concept.id == '1745121':		# wiki page for test
                Settings.logger.debug("Concept CONLLU: '" + concept.features.conllu + "'")
                Settings.logger.debug("Parsed CONLLU: '" + str(concept.features.numberOfSentences) + "'")

        '''
        i = 0
        for concept in MyModel.dataset:
            if concept.id == '1745121':
                print( "concept content is: " + concept.content)
            concept.features.sentences = nltk.tokenize.sent_tokenize(concept.content)
            concept.features.words = nltk.word_tokenize(concept.content)

            concept.features.posTags = nltk.pos_tag(concept.features.words) # Part of speech tags
            fd = nltk.FreqDist(concept.features.posTags)    # analisi frequenziale dei pos tags
            concept.features.entities = nltk.chunk.ne_chunk(concept.features.posTags)   # estrazione entities dal pos tag
            # Just for quick review, to delete
            if concept.id == '1745121':
                print(fd['NN'] / len(concept.features.words))   # mostra percentuale nomi nel concetto
                print('concept words: ' + str(concept.features.words))
                print('concept pos tags: ' + str(concept.features.posTags))
                print('concept entities: ' + str(concept.features.entities))
                print('concept sentences: ' + str(concept.features.sentences))
        '''

