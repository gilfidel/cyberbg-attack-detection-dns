import sys
import os
import logging

import inspect
import six
import argh
import pathlib2
import pandas

import DNSPreprocessor
import utils

LOG = logging.getLogger(__name__)


def preprocess_file(file_name: str, target_file_name: str):
    LOG.info(f'Preprocessing {file_name} => {target_file_name}')
    df: pandas.DataFrame = DNSPreprocessor.preprocess_client_dsp_data(file_name)
    df.to_csv(target_file_name, ',', index=False)


def preprocess(data_dir_name: str, out_dir_name: str):
    data_dir = pathlib2.Path(data_dir_name)
    out_dir = pathlib2.Path(out_dir_name)

    out_dir.mkdir(parents=True, exist_ok=True)

    for file_path in data_dir.glob('dnsSummary_user*.pcap.csv'): #dnsSummary_user292.pcap.csv
        preprocess_file(str(file_path), str(out_dir.joinpath(f'{file_path.name}-proc.csv')))


def _main():
    utils.init_logging(r'logs\main.log')

    out = six.StringIO()
    # Expose all functions that don't begin with an underscore "_" in the current module
    argh.dispatch_commands(
        [obj for name, obj in inspect.getmembers(sys.modules[__name__]) if
         inspect.isfunction(obj) and obj.__module__ == '__main__' and not name.startswith('_')],
        output_file=out
    )

    print(out.getvalue())


if '__main__' == __name__:
    _main()
