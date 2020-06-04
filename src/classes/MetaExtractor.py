from Model import Model
import wikipediaapi    # https://github.com/martin-majlis/Wikipedia-API/
# Not using official library because of https://stackoverflow.com/questions/34869597/wikipedia-api-for-python
from Settings import Settings

class MetaExtractor:

    def __init__(self):
        self.wikipedia = wikipediaapi.Wikipedia(
            language='it',
            extract_format=wikipediaapi.ExtractFormat.WIKI) # .HTML for the marked-up version

    def print_sections(self, sections, level=0):
        for s in sections:
            print("%s: %s - %s" % ("*" * (level + 1), s.title, s.text[0:40]))
            self.print_sections(s.sections, level + 1)

    def print_links(self, page):    # Print Links To Other Pages
        links = page.links
        for title in sorted(links.keys()):
            print("%s: %s" % (title, links[title]))

    def print_categories(self, page):     # Prints Categories
        categories = page.categories
        for title in sorted(categories.keys()):
            print("%s: %s" % (title, categories[title]))


    def annotateConcepts(self):
        for concept in Model.dataset:
            # NOTE page contains backlinks: links from another page that points to page
            # (e.g. page='indice di rifrazion, another page=anisotropia)
            page = self.wikipedia.page(concept.title)
            if concept.id != str(page.pageid):
                raise ValueError("Couldn't find a correspondence in wikiApi for concept '" +
                                 concept.title + "' with Id: '" + concept.id + "'")
            concept.meta = page


    def links(self):
        for conceptA in Model.dataset:
            for conceptB in Model.dataset:
                if conceptB.title in conceptA.meta.links:
                    pass    # TODO: register this to the PairFeatures class

    def showConcept(self, page):
        print(page.title)
        self.print_sections(page.sections)