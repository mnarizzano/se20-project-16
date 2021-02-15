import logging

class Settings:
    """Configuration module

    This class is fully static and needs no instance. All values are global
    Here are defined all parameters of the run like paths, feqtures to use, etc
    All parameters are set to a default but can be changed at runtime (e.g. by a GUI)
    to interactively configure the run of the program.

    Attributes:
        resourcePath: root folder for the execution, other locations are relative to this
        resultFolder: path to output folder
        baseFolder: path to folder containing single file input dataset (rev 2)
        datasetPath: path to folder containing dataset in split files (rev 1)
        testsetPath: path to folder containing the file of pairs to be classified
        conceptsPickle: path for the cached dataset
        pairFeaturesPickle: path of the cached pairFeatures object
        pairsPickle: path of the cached desiredGraph object
        resampleSmallerClass: bool indicating whether or not to apply weight to rebalance dataset
        guidePage: location of the HTML guide page shown in the UI
        savedConfigurations: path of the file containing saved configurations
        logger: singleton instance of the logger used throughout the program
        neurons = number of neurons in each layer of the NN
        layers = number of layers of the NN
        kfoldSplits = Number of Folds into which split the dataset
        epoch = Number of epochs on which train the Classifier
        numberOfTopics = LDA configuration parameter: number of hidden topics to look for.
    """

    # Paths
    resourcePath = '../resources/'
    resultFolder = resourcePath + 'results/'
    baseFolder = resourcePath + 'dataset/single_file/'
    datasetPath = baseFolder + 'PRELEARN_training_data'
    testsetPath = baseFolder + 'PRELEARN_test_data'   # set this to None to skip prediction
    conceptsPickle = resourcePath + 'cached/concepts.pickle'
    pairFeaturesPickle = resourcePath + 'cached/pairsFeatures.pickle'
    pairsPickle = resourcePath + 'cached/prereqPairs.pickle'

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
