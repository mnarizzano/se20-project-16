import nltk
from ufal.udpipe import Model, Pipeline, ProcessingError
from Parser import Parser

class FeatureExtractor:

    udpipeModelPath = '../resources/Model/italian-isdt-ud-2.5-191206.udpipe'

    def extractNouns(self):
        udpipeModel = Model.load(self.udpipeModelPath)
        pipeline = Pipeline(udpipeModel, 'tokenize',
                            Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        error = ProcessingError()

        for concept in Parser.concepts:
            processedConcept = pipeline.process(concept.content, error)
            if concept.id == '1745121':		# pagina wiki per esempio uso udpipe
                print(processedConcept)

        i = 0
        for concept in Parser.concepts: #TODO concepts should be in a dedicated class and not in Parser
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
