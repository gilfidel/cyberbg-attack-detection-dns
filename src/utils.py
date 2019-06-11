import sys
import logging
import logging.config

import pathlib2


def init_logging(log_file_name):
    pathlib2.Path(log_file_name).parent.mkdir(exist_ok=True)
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'default': {'format': '%(asctime)s [%(name)-10s] %(levelname)-5s\t%(message)s',
                        'datefmt': '%Y-%m-%d_%H-%M-%S'}
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'default',
                'filename': log_file_name,
                'maxBytes': 10 * 1024 * 1024,
                'backupCount': 3
            }
        },
        'loggers': {
            '': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': True
            }
        },
        'disable_existing_loggers': False
    })
    logging.getLogger('parso').setLevel(logging.ERROR)
    logging.getLogger('main').info(f'=== STARTED: {sys.argv}')
