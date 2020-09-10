import logging

class Settings:
    # Paths
    resourcePath = '../resources/'
    datasetPath = resourcePath + 'dataset/single_file/PRELEARN_training_data'
    #datasetPath = resourcePath + 'dataset/split_files/sample'
    conceptsPickle = resourcePath + 'cached/concepts.pickle'
    pairsPickle = resourcePath + 'cached/pairs.pickle'

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