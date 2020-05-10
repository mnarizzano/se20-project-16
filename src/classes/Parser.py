import os
import pickle
import xml.etree.ElementTree as ET
import urllib.parse
import re

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


    def parseFile(self, file):
        domain = file.name.split('-')[0]
        fileString = open(file, 'r').read()

        # check if < and > are part of XML or text
        # original regex (?<!([\n]))[<](?!(doc|/doc))
        isNotInitTag = re.compile("(?<!([\\n]))[<](?!(doc |/doc>))")  # returns true if < is not XML
        isNotClosingTag = re.compile("(?<!([\"|</doc]))[>](?!(\\n))")    # returns true if > is not of the XML

        # need to encode casual '<' or '>' since they break XML parsing
        lessThan = urllib.parse.quote('<')
        moreThan = urllib.parse.quote('>')
        part = fileString[fileString.find('e altre 288 possibili super Terre')-50:fileString.find('e altre 288 possibili super Terre')+100]
        print(part)

        # find not XML < and > and encode them
        fileString = isNotInitTag.sub(lessThan, fileString)
        fileString = isNotClosingTag.sub(moreThan, fileString)
        part = fileString[fileString.find('e altre 288 possibili super Terre') - 50:fileString.find(
            'e altre 288 possibili super Terre') + 100]
        print(part)
        fileString = '<root>' + fileString + '</root>'
        tree = ET.fromstring(fileString)
        #root = tree.getroot()
        for child in tree:
            info = child[1].attrib
            #c = Concept(id = info['id'], url = info['url'], title = info['title'], content = info['text'])
            print(child)
        a = 26


    def parseMeta(self):
        pass