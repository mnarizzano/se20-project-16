import logging
from Parser import Parser
from Engine import Engine

global concepts
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('myLogger')
    p = Parser(logger)
    concepts = p.parse()
    logger.info('Finished Parsing')
    engine = Engine()
    engine.process()