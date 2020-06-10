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
    resourcePath = Settings.resourcePath
    conceptsPickle = Settings.conceptsPickle
    pairsPickle = Settings.pairsPickle
    force = False
    pairs = None


    def listDirectory(self, path):
        if os.path.exists(path):
            return os.scandir(path)


    def parse(self, force = False):
        if force:
            if os.path.exists(self.conceptsPickle):
                os.remove(self.conceptsPickle)
            if os.path.exists(self.pairsPickle):
                os.remove(self.pairsPickle)

        # Load Concepts or parse them from "*-pages.txt" files
        if os.path.exists(self.conceptsPickle):
            with open(self.conceptsPickle, 'rb', encoding='utf8') as file:
                self.concepts = pickle.load(file)
        else:
            for entry in self.listDirectory(self.resourcePath):
                if entry.is_file() and entry.name.split('-')[-1] == 'pages.txt':
                    self.parsePages(entry)

        # Load Pairs from pickle or parse them from "*-pairs.txt" files
        if os.path.exists(self.pairsPickle):
            with open(self.pairsPickle, 'rb', encoding='utf8') as file:
                self.pairs = pickle.load(file)
        else:
            Model.desiredGraph = GraphMatrix()
            for entry in self.listDirectory(self.resourcePath):
                if entry.is_file() and entry.name.split('-')[-1] == 'pairs.txt':
                    self.parsePairs(entry)
        Model.desiredGraph.plotGraph()

    def parsePages(self, file):
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
            content = child.text.replace(urllib.parse.quote('<'), '<').replace(urllib.parse.quote('>'), '>')
            c = Concept(id=id, url=url, title=title, content=content, domain=domain)
            Model.dataset.append(c)

    def parsePairs(self, entry):
        file = open(entry, 'r', encoding='utf8')
        for line in file.readlines():
            Settings.logger.debug("pairs line fetched: " + line)
            prereq = line.strip('\n')
            if len(prereq) > 1:
                prereq = prereq.split(',')
                Model.desiredGraph.addPrereq(prereq[0], prereq[1], prereq[2])
