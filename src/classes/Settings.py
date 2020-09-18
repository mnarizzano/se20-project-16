import logging

class Settings:
    # Paths
    resourcePath = '../resources/'
    datasetPath = resourcePath + 'dataset/single_file/PRELEARN_training_data'
    #datasetPath = resourcePath + 'dataset/split_files/sample'
    Models = resourcePath + 'Model/'
    glove_WIKI = Models + 'Glove/glove_WIKI'
    conceptsPickle = resourcePath + 'cached/concepts.pickle'
    pairFeaturesPickle = resourcePath + 'cached/pairsFeatures.pickle'
    pairsPickle = resourcePath + 'cached/prereqPairs.pickle'

    # Caching
    useCache = True

    # Udpipe
    udpipeModelPath = '../resources/Model/italian-isdt-ud-2.5-191206.udpipe'

    # Word-Embedding model path
    wordVecModelPath = '../resources/Model/italian-isdt-ud-2.5-191206.udpipe'

    # Logging
    logger = None
    logLevel = logging.DEBUG
    logging.basicConfig(level=logLevel,
                        format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('myLogger')