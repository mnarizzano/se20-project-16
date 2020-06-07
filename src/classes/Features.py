
class Features:
    conllu = None
    sentences = []
    annotatedSentences = None # This contains a list where each entry is an annotated sentence of the concept

    def get_numberOfSentences(self):
        return len(self.sentences)

    def get_annotatedSentences(self):
        return self.sentences