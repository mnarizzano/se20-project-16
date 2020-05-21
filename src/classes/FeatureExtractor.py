import nltk
from Parser import Parser

class FeatureExtractor:
    global concepts

    def extractNouns(self):
        i=0
        for concept in Parser.concepts: #TODO concepts should be in a dedicated class and not in Parser
            #if concept.id == '1745121':
            print( "concept content is: " + concept.content)
            #concept.features.tokens = nltk.word_tokenize(str(concept.content))
            # Just for quick review, to delete
            if concept.id == '1745121':
                print(concept.features.tokens)