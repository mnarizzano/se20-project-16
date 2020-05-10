import logging
from Parser import Parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('myLogger')
    p = Parser(logger)
    p.parse()