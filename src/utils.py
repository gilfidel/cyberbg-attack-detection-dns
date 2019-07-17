import sys
import datetime
import time
import logging
import logging.config

import pathlib


def init_logging(log_file_name):
    pathlib.Path(log_file_name).parent.mkdir(exist_ok=True)
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

DEFAULT_STRFTIME_FORMAT = '%Y-%m-%d_%H-%M-%S'
DATE_STRFTIME_FORMAT = '%Y-%m-%d'

def get_time_interval_str( seconds, strip_milliseconds = True ):
    if strip_milliseconds:
        seconds = int( seconds )
    return str( datetime.timedelta( seconds = seconds ) )

def get_time_stamp( time_t, gmt_timestamp = False, with_micro_seconds = False, strftime_format = DEFAULT_STRFTIME_FORMAT ):
#    tm = time.localtime( time_t )
    if gmt_timestamp:
        dt = datetime.datetime.utcfromtimestamp( time_t )
    else:
        dt = datetime.datetime.fromtimestamp( time_t )

    if with_micro_seconds:
        strftime_format += '.%f'
    return dt.strftime( strftime_format )


def get_current_time_stamp( gmt_timestamp = False, with_micro_seconds = False ):
    return get_time_stamp( time.time(), gmt_timestamp, with_micro_seconds  )
