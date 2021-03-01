__author__ = "Parizzi Andrea"
__license__ = "Public Domain"
__version__ = "1.0"

import os
import pickle
import xml.etree.ElementTree as ET
import urllib.parse
import re
from Settings import Settings
from Concept import Concept
from GraphMatrix import GraphMatrix
from Model import Model

class Parser(object):
    """ Parser to read inputFiles, write calculated labels to outputFiles and cache datasets
    """
    resourcePath = Settings.resourcePath
    conceptsPickle = Settings.conceptsPickle
    pairsPickle = Settings.pairsPickle
    test = {}

    def listDirectory(self, path):
        """List items in a directory

        Args:
            path: string containing the query path
        Returns: [string] list of items in the folder
        """
        if os.path.exists(path):
            return os.scandir(path)

    def cache(self):
        """Caches the Dataset, including parsed Concepts and all calculated single features
        """
        Settings.logger.debug('Caching dataset...')
        pickle.dump(Model.dataset, open(self.conceptsPickle, "wb+"))

    def parseTest(self):
        """For each file named *pairs in the path specified by the Settings module
           parses it and assigns it to a Dictionary indexed by the name of the file
        """
        numberOfEntries = 0
        for entry in self.listDirectory(Settings.testsetPath):
            if entry.is_file() and entry.name.__contains__('pairs'):
                parsed = self.parseTestFile(entry)
                numberOfEntries = numberOfEntries + len(parsed)
                self.test[entry.name.split('-')[0]] = parsed

    def parseTestFile(self, entry):
        """Parse a single *pairs file, formatted as Documentation

        Args:
            entry: path to the pairs file
        Returns: [[string, string], [], ...] list of pairs of Concepts titles
        """
        parsed = []
        file = open(entry, 'r', encoding='utf8')
        for line in file.readlines():
            Settings.logger.debug("test pairs line fetched: " + line)
            prereq = line.strip('\n')
            if len(prereq) > 1:
                prereq = prereq.split(',')
                fromConcept = prereq[0] # Model.dataset[Model.dataset.index(prereq[0])]
                toConcept = prereq[1] # Model.dataset[Model.dataset.index(prereq[1])]
                parsed.append([fromConcept, toConcept])
        return parsed

    def parse(self, cache=Settings.useCache):
        """Parses the Concept Dataset and the groundtruth labels

        Checks if cached Dataset and GraphMatrix exist. If they dont cycles
        through all files and if their name correspond to the Documentation name for
        the Dataset file or the GraphMatrix file parses them.
        Assigns Model.dataset and Model.desiredGrph in the process
        """
        if not cache:
            if os.path.exists(self.conceptsPickle):
                os.remove(self.conceptsPickle)
            if os.path.exists(self.pairsPickle):
                os.remove(self.pairsPickle)

        # Load Concepts or parse them from "*-pages.txt" files
        if os.path.exists(self.conceptsPickle):
            with open(self.conceptsPickle, 'rb') as file:
                Model.dataset = pickle.load(file)
        else:
            for entry in self.listDirectory(Settings.datasetPath):
                if entry.is_file() and entry.name.__contains__('pages'):
                    if Settings.datasetPath.__contains__('split_files'):
                        self.parsePages(entry)
                    else:
                        self.parseSinglePage(entry)

        # Load desired Pairs from pickle or parse them from "*-pairs.txt" files
        if os.path.exists(self.pairsPickle):
            with open(self.pairsPickle, 'rb') as file:
                Model.desiredGraph = pickle.load(file)
        else:
            domains = []
            for entry in self.listDirectory(Settings.datasetPath):
                if entry.is_file() and entry.name.__contains__('pairs'):
                    domains.append(entry.name.split('-')[0])
            Model.desiredGraph = GraphMatrix(domains)
            numberOfEntries = 0
            for entry in self.listDirectory(Settings.datasetPath):
                if entry.is_file() and entry.name.__contains__('pairs'):
                    numberOfEntries = numberOfEntries + self.parsePairs(entry)
            Settings.logger.debug("Loaded " + str(numberOfEntries) + " entries of prerequisite relationship")
        Model.desiredGraph.plotPrereqs()

    def parseSinglePage(self, file):
        """XML parsing of dataset when contained in a single file (rev 2 of the datset input file format)
        """
        domain = 'single_file_pages_has_no_domain'
        fileString = open(file, 'r', encoding='utf8').read()

        tree = ET.fromstring(fileString)
        # root = tree.getroot()
        for doc in tree:
            info = doc.attrib
            id = info['id']
            url = info['url']
            for bodyChild in doc:
                if bodyChild.tag == 'title':
                    title = bodyChild.text
                elif bodyChild.tag == 'text':
                    content = bodyChild.text
            if not title in Model.dataset:
                c = Concept(id=id, url=url, title=title, content=content, domain=domain)
                Model.dataset.append(c)
            else:
                Settings.logger.info("Skipping concept " + title + " because it is already present")

    def parsePages(self, file):
        """XML parsing of the part of dataset related to a given domain (first version of the dataset file)
        """
        domain = file.name.split('-')[0]
        fileString = open(file, 'r', encoding='utf8').read()

        # True if < and > are part of text (i.e. not-XML-related '<' and '>')
        isNotInitTag = re.compile("(?<!([\\n]))[<](?!(doc |/doc>))")  # returns true if < is not XML
        isNotClosingTag = re.compile("(?<!([\"|</doc]))[>](?!(\\n))")    # returns true if > is not of the XML

        # need to encode not-XML-related '<' or '>' since they break XML parsing
        lessThan = urllib.parse.quote('<')
        moreThan = urllib.parse.quote('>')

        # find not XML '<' and '>' and replace them
        fileString = isNotInitTag.sub(lessThan, fileString)
        fileString = isNotClosingTag.sub(moreThan, fileString)

        # create wrapping XML tag
        fileString = '<root>' + fileString + '</root>'
        tree = ET.fromstring(fileString)
        #root = tree.getroot()
        for child in tree:
            info = child.attrib
            id = info['id']
            url = info['url']
            title = info['title']
            # to avoid having duplicated concepts from different domains
            # which in turn lead to considering a single prereq relationship multiple times and biases the model
            # TODO if input is free from duplicates this is just overhead
            if not title in Model.dataset:
                content = child.text.replace(urllib.parse.quote('<'), '<').replace(urllib.parse.quote('>'), '>')
                c = Concept(id=id, url=url, title=title, content=content, domain=domain)
                Model.dataset.append(c)

    def parsePairs(self, entry):
        """Parser for the desired labels file, builds the Model.desiredGraph
        """
        numberOfEntries = 0
        file = open(entry, 'r', encoding='utf8')
        for line in file.readlines():
            numberOfEntries += 1
            Settings.logger.debug("pairs line fetched: " + line)
            prereq = line.strip('\n')
            if len(prereq) > 1:
                prereq = prereq.split(',')
                Model.desiredGraph.addPrereq(prereq[0], prereq[1], prereq[2], entry.name.split('-')[0])
        return numberOfEntries