__author__ = "Moggio Alessio, Parizzi Andrea"
__license__ = "Public Domain"
__version__ = "1.0"

from Model import Model
import wikipediaapi    # https://github.com/martin-majlis/Wikipedia-API/
from Settings import Settings
import pickle
from Parser import Parser

class MetaExtractor:
    """Utility module to extract structured information from wikipedia pages

    ...

    Attributes
    ----------
    pairFeatures : PairFeatures
        holds all features related to a pair of Concepts
    parser : Parser
        own instance of the Parser

    """
    pairFeatures = None
    parser = Parser()

    def __init__(self, pairFeatures):
        """Constructor
        Instantiates the object to query and dump Wikipedia
        """
        self.wikipedia = wikipediaapi.Wikipedia(
            language='it',
            extract_format=wikipediaapi.ExtractFormat.WIKI)  # .HTML for the marked-up version
        self.pairFeatures = pairFeatures

    def print_sections(self, sections, level=0):
        """Prints all sections of the Concept wikipedia page
        """
        for s in sections:
            print("%s: %s - %s" % ("*" * (level + 1), s.title, s.text[0:40]))
            self.print_sections(s.sections, level + 1)

    def print_links(self, page):
        """Prints all outgoing links of the Concept wikipedia page
        """
        links = page.links
        for title in sorted(links.keys()):
            print("%s: %s" % (title, links[title]))

    def print_categories(self, page):
        """Prints all categories of the Concept wikipedia page
        """
        categories = page.categories
        for title in sorted(categories.keys()):
            print("%s: %s" % (title, categories[title]))

    def annotateConcepts(self):
        """Downloads Wikipedia info for each Concept and stores it in the Concept.meta attributes

        Also already fills a few meta fields for the concepts, saves the dataset to disk at the end
        """
        Settings.logger.debug("annotateConcepts")
        loaded = (Model.dataset[len(Model.dataset) - 1].meta.links is not None) and \
                 (Model.dataset[len(Model.dataset) - 1].meta.links != []) and \
                 (Model.dataset[len(Model.dataset) - 1].meta.categories is not None) and \
                 (Model.dataset[len(Model.dataset) - 1].meta.categories != [])
        if not loaded:  # if already present it has been loaded from pickle
            for concept in Model.dataset:
                page = self.wikipedia.page(concept.title)
                if concept.id != str(page.pageid):
                    raise ValueError("Couldn't find a correspondence in wikiApi for concept '" +
                                     concept.title + "' with Id: '" + concept.id + "'")
                concept.meta = page
            self.parser.cache()

    def extractLinks(self):
        """Extract outgoing links to a dedicated list
        """
        for concept in Model.dataset:
            for title in concept.meta.links.keys():
                concept.meta.links.append(title)

    def extractCategories(self):
        """Extract categories to a dedicated list
        """
        for concept in Model.dataset:
            for title in concept.meta.categories.keys():
                concept.meta.categories.append(
                    title.removeprefix('Categoria:'))

    def showConcept(self, page):
        """Logger method to show fetched infos
        """
        print(page.title)
        self.print_sections(page.sections)

    def cache(self):
        """saves the pairFeatures object to Disk
        """
        Settings.logger.debug('Caching pairFeatures...')
        pickle.dump(self.pairFeatures, open(Settings.pairFeaturesPickle, "wb+"))

    def extractLinkConnections(self):
        """For each Concept pair check if a direct link exists between them
        """
        Settings.logger.debug("extractLinkConnections")
        if not self.pairFeatures.linksLoaded():
            for concept in Model.dataset:
                concept.features.totalIncomingLinks = 0
            for conceptA in Model.dataset:
                conceptA.features.totalOutgoingLinks = len(conceptA.meta.links.keys())
                for conceptB in Model.dataset:
                    for value in conceptA.meta.links.keys():
                        # links from A
                        if conceptB.title == value:
                            conceptB.features.totalIncomingLinks = conceptB.features.totalIncomingLinks + 1
                            self.pairFeatures.addLink(conceptA, conceptB, 1)
            self.cache()
        else:
            Settings.logger.debug("Skipping extractLinkConnection because it was cached")

    def referenceDistance(self):
        """Calculates the RefD metric
        """
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
