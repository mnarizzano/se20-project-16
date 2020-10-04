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
    if Settings.getPredictions:
        for mode in [True, False]:
            for featTypes in [True, False]:
                Settings.CrossDomain = mode
                Settings.useRefD = featTypes
                # Calculate Engine Performance
                engine = Engine()
                result = engine.process() # might be cv results or testSet predictions, depending on Settings.generateOutput'
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
    else:
        engine = Engine()
        result = engine.process()  # might be cv results or testSet predictions, depending on Settings.generateOutput'
    k = 'debug breakpoint'

    if Settings.useCache:
        p.cache()