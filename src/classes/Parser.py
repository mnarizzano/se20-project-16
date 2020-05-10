import os
import pickle
import xml.etree.ElementTree as ET
import urllib.parse
import re
from Concept import Concept

class Parser(object):
    resourcePath = '../resources/'
    dataPath = resourcePath + 'dataset.pickle'
    logger = None
    force = False
    concepts = []

    def __init__(self, logger):
        self.logger = logger


    def listDirectory(self, path):
        if os.path.exists(path):
            return os.scandir(path)


    def parse(self, force = False):
        if os.path.exists(self.dataPath):
            if force:
                os.remove(self.dataPath)
            else:
                with open(self.dataPath, 'rb') as file:
                    self.concepts = pickle.load(file)
                return self.concepts
        for entry in self.listDirectory(self.resourcePath):
            if entry.is_file() and entry.name.split('-')[-1] == 'pages.txt':
                self.parseFile(entry)
        for concept in self.concepts:
            self.parseMeta(concept)     # TODO NOT SURE I CAN EDIT class concepts if using this iterator, maybe duplicate concepts for lacal editingand overwrite class concepts
        return self.concepts

    def parseFile(self, file):
        domain = file.name.split('-')[0]
        fileString = open(file, 'r').read()

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
            self.concepts.append(c)

    def parseMeta(self, concept):
        pass