from Parser import Parser
from Engine import Engine
from Model import Model
from Settings import Settings

if __name__ == '__main__':
    # Parse files in Specified folder, optionally we can add input to modify Settings.resourcePath
    p = Parser()
    p.parse()
    p.parseTest()
    Settings.logger.info('Finished Parsing')

    # Calculate Baseline Performance
    '''
    base = Baseline()
    basePerformance = base.process()
    '''
    if Settings.fullRun:
        for mode in [True, False]:
            for featTypes in [True, False]:
                Settings.CrossDomain = mode
                Settings.useRefD = featTypes
                # Calculate Engine Performance
                engine = Engine()
                result = engine.process() # might be cv results or testSet predictions, depending on Settings.generateOutput'
                if Settings.getPredictions:
                    if Settings.CrossDomain:
                        fileName = Settings.resultFolder + 'cross/'
                    else:
                        fileName = Settings.resultFolder + 'in/'
                    if Settings.useRefD:
                        fileName = fileName + 'meta/'
                    else:
                        fileName = fileName + 'raw/'
                    for domain in result['result'].keys():
                        file = open(fileName + str(domain), 'w')
                        for element in result['result'][domain]:
                            file.write(str(element[2]) + '\n')
                        file.close()
                if Settings.manualCV:   # engine returned probabilities distributions, write them to file
                    if Settings.CrossDomain:
                        fileName = Settings.resultFolder + 'CV/cross_'
                    else:
                        fileName = Settings.resultFolder + 'CV/in_'
                    if Settings.useRefD:
                        fileName = fileName + 'meta_'
                    else:
                        fileName = fileName + 'raw_'
                    file = open(fileName + 'probabilities', 'w')
                    for probability in result['correctProbabilities'].keys():
                        file.write(str(probability) + ', ' +
                                   str(result['correctProbabilities'][probability]) + ', ' +
                                   str(result['wrongProbabilities'][probability]) + '\n')
                    file.close()
    else:
        engine = Engine()
        result = engine.process()  # might be cv results or testSet predictions, depending on Settings.generateOutput'
    k = 'debug breakpoint'

    if Settings.useCache:
        p.cache()