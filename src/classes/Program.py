from Parser import Parser
from Engine import Engine
from Model import Model
from Settings import Settings

if __name__ == '__main__':
    # Parse files in Specified folder, optionally we can add input to modify Settings.resourcePath
    p = Parser()
    p.parse()
    Settings.logger.info('Finished Parsing')

    # Calculate Baseline Performance
    '''
    base = Baseline()
    basePerformance = base.process()
    '''

    # Calculate Engine Performance
    engine = Engine()
    engine.process()
    # plot statistics
    engine.plot()

    if Settings.useCache:
        p.cache()