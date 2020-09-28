from Model import Model
import wikipediaapi    # https://github.com/martin-majlis/Wikipedia-API/
# Not using official library because of https://stackoverflow.com/questions/34869597/wikipedia-api-for-python
from Settings import Settings
import pickle
from Parser import Parser


class MetaExtractor:

    pairFeatures = None
    parser = Parser()

    def __init__(self, pairFeatures):
        self.wikipedia = wikipediaapi.Wikipedia(
            language='it',
            extract_format=wikipediaapi.ExtractFormat.WIKI)  # .HTML for the marked-up version
        self.pairFeatures = pairFeatures

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
        Settings.logger.debug("annotateConcepts")
        loaded = (Model.dataset[len(Model.dataset) - 1].meta.links is not None) and \
                 (Model.dataset[len(Model.dataset) - 1].meta.links != []) and \
                 (Model.dataset[len(Model.dataset) - 1].meta.categories is not None) and \
                 (Model.dataset[len(Model.dataset) - 1].meta.categories != [])
        if not loaded:  # if already present it has been loaded from pickle
            for concept in Model.dataset:
                # NOTE page contains backlinks: links from another page that points to page
                # (e.g. page='indice di rifrazion, another page='anisotropia')
                page = self.wikipedia.page(concept.title)
                if concept.id != str(page.pageid):
                    raise ValueError("Couldn't find a correspondence in wikiApi for concept '" +
                                     concept.title + "' with Id: '" + concept.id + "'")
                concept.meta = page
            self.parser.cache()

    def extractLinks(self):
        for concept in Model.dataset:
            for title in concept.meta.links.keys():
                concept.meta.links.append(title)

    def extractCategories(self):
        for concept in Model.dataset:
            for title in concept.meta.categories.keys():
                concept.meta.categories.append(
                    title.removeprefix('Categoria:'))

    def showConcept(self, page):
        print(page.title)
        self.print_sections(page.sections)

    def cache(self):
        Settings.logger.debug('Caching pairFeatures...')
        pickle.dump(self.pairFeatures, open(Settings.pairFeaturesPickle, "wb+"))

    def extractLinkConnections(self):
        Settings.logger.debug("extractLinkConnections")
        if not self.pairFeatures.linksLoaded():
            for conceptA in Model.dataset:
                for conceptB in Model.dataset:
                    for value in conceptA.meta.links.keys():
                        if conceptB.title == value:
                            self.pairFeatures.addLink(conceptA, conceptB, 1)
            self.cache()
        else:
            Settings.logger.debug("Skipping extractLinkConnection because it was cached")

    def referenceDistance(self):  # using EQUAL weights
        if not self.pairFeatures.RefDLoaded():
            Settings.logger.debug('Calculating Reference Distance')
            for conceptA in Model.dataset:
                for conceptB in Model.dataset:
                    num1 = 0
                    num2 = 0
                    den1 = 0
                    den2 = 0
                    for concept in Model.dataset:
                        num1 += (self.pairFeatures.pairFeatures[concept.id][conceptB.id].link *
                                 self.pairFeatures.pairFeatures[conceptA.id][concept.id].link)
                        num2 += (self.pairFeatures.pairFeatures[concept.id][conceptA.id].link *
                                 self.pairFeatures.pairFeatures[conceptB.id][concept.id].link)
                        den1 += (self.pairFeatures.pairFeatures[conceptA.id][concept.id].link)
                        den2 += (self.pairFeatures.pairFeatures[conceptB.id][concept.id].link)
                    if (den1 != 0 and den2 != 0):
                        dist = (num1 / den1) - (num2 / den2)
                        self.pairFeatures.setReferenceDistance(conceptA, conceptB, dist)
                    else:
                        # if den1 or den2 = 0, it means that A and B are no prerequisites
                        self.pairFeatures.setReferenceDistance(conceptA, conceptB, 0)
            self.cache()
            Settings.logger.debug('Finished calculating RefD')
        else:
            Settings.logger.debug('Skipping referenceDistance cause it was cached')
