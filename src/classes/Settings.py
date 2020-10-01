import logging

class Settings:
    # Paths
    resourcePath = '../resources/'
    datasetPath = resourcePath + 'dataset/single_file/PRELEARN_training_data'
    testsetPath = resourcePath + 'dataset/single_file/PRELEARN_test_data'
    #datasetPath = resourcePath + 'dataset/split_files/sample'
    Models = resourcePath + 'Model/'
    glove_WIKI = Models + 'Glove/glove_WIKI'
    conceptsPickle = resourcePath + 'cached/concepts.pickle'
    pairFeaturesPickle = resourcePath + 'cached/pairsFeatures.pickle'
    pairsPickle = resourcePath + 'cached/prereqPairs.pickle'

    # type of run
    generateOutput = False

    # type of balancing algorithm
    resampleSmallerClass = True

    # paths for gui
    guidePage = '../resources/guide.html'
    savedConfigurations = '../resources/saved.txt'

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

    # Inserted value for NN
    neurons = 0
    layers = 0
    kfoldSplits = 0
    epoch = 0

    # Checkboxes for feature selection
    useRefD = True
    useConceptLDA = True
    useJaccard = True
    useContainsLink = True
    useLDACrossEntropy = True
    useLDA_KLDivergence = True
    contains = True

    # lemmatization settings
    useNouns = True
    useVerbs = False
    useAdjectives = False

    # set run mode
    CrossDomain = True  # if False trains on all domains
