import logging

class Settings:
    # Paths
    resourcePath = '../resources/'
    resultFolder = resourcePath + 'results/'
    baseFolder = resourcePath + 'dataset/single_file/'
    datasetPath = baseFolder + 'PRELEARN_training_data'
    testsetPath = baseFolder + 'PRELEARN_test_data'   # set this to None to skip prediction
    #datasetPath = resourcePath + 'dataset/split_files/sample'
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

    # Word-Embedding model path
    Models = resourcePath + 'Model/'
    glove_WIKI = Models + 'Glove/glove_WIKI'
    # Udpipe model for lemmatization
    udpipeModelPath = Models + 'italian-isdt-ud-2.5-191206.udpipe'

    # Logging
    logger = None
    logLevel = logging.DEBUG
    logging.basicConfig(level=logLevel,
                        format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('myLogger')

    # Inserted value for NN
    neurons = 25
    layers = 1
    kfoldSplits = 2
    epoch = 20

    # LDA
    numberOfTopics = 10

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
    CrossDomain = False  # if False trains on all domains

    # chose which type of CV run
    manualCV = True
    crossValidateCV = False  # ie using scikit method cross_validate

    # chose if run predictions or CV only
    getPredictions = False
    fullRun = True
