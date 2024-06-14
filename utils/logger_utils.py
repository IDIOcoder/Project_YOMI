import json
import logging
import logging.config

with open('utils/logger.json', 'r') as f:
    config = json.load(f)
logging.config.dictConfig(config)

DEFAULT = logging.getLogger('default')
CHATBOT = logging.getLogger('chatbot')


def get_logger(name):
    return logging.getLogger(name)
