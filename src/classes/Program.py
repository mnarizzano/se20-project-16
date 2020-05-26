from Parser import Parser
from Engine import Engine
from Model import Model
from Settings import Settings

if __name__ == '__main__':
    # Parse files in Specified folser, optionally add input to modify Settings.resourcePath
    p = Parser()
    p.parse()
    Settings.logger.info('Finished Parsing')
    # Set parsed concepts and prerequisites matrix
    Model.desiredGraph = p.pairs
    Model.dataset = p.concepts

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