import logging

class Settings:
    # Paths
    resourcePath = '../resources/'
    conceptsPickle = resourcePath + 'cached/concepts.pickle'
    pairsPickle = resourcePath + 'cached/pairs.pickle'
    
    savedConfigurations = '../resources/saved.txt'

    # Caching
    useCache = True

    # Udpipe
    udpipeModelPath = '../resources/Model/italian-isdt-ud-2.5-191206.udpipe'

    # Logging
    logger = None
    logLevel = logging.DEBUG
    logging.basicConfig(level=logLevel,
                        format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('myLogger')