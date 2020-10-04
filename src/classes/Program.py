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

    # Calculate Engine Performance
    engine = Engine()
    result = engine.process() # might be cv results or testSet predictions, depending on Settings.generateOutput
    # plot statistics
    k = 'debug breakpoint'

    if Settings.useCache:
        p.cache()