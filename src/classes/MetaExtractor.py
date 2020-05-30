from Model import Model
import wikipedia    # https://wikipedia.readthedocs.io/en/latest/code.html
# Not using official library because of https://stackoverflow.com/questions/34869597/wikipedia-api-for-python
from Settings import Settings

class MetaExtractor:

    def __init__(self):
        wikipedia.set_lang("it")


    def annotateConcepts(self):
        for concept in Model.dataset:
            # page.content shows the root structure of page sections, which are nested with an increasing number of "="
            page = wikipedia.page(pageid=concept.id)
            sections = page.sections
            section = page.section(sections[0])
            if concept.title != page.title:
                raise ValueError("Couldn't find a correspondence in wikiApi for concept '" +
                                 concept.title + "' with Id: '" + concept.id + "'")