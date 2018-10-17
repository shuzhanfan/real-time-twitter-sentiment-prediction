from ConfigParser import SafeConfigParser
import logging

#This method will read and parse a config file and return a parser object.
def _get_config():
    parser = SafeConfigParser()
    parser.read('config')
    return parser

#This method will create a logger object and set the level to info.
def _logger():
    #create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    #create a formatter
    formatter = logging.Formatter('%(message)s')
    #add formatter to console handler
    handler.setFormatter(formatter)
    #add console handler to logger
    logger.addHandler(handler)
    return logger
