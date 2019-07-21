import sys
import logging

import inspect
import six
import argh
import pathlib
import pandas

import query_name_learner
import DNSPreprocessor
import data_loader
import utils

LOG = logging.getLogger(__name__)


def preprocess_file(file_name: str, target_file_name: str):
    LOG.info(f'Preprocessing {file_name} => {target_file_name}')
    df: pandas.DataFrame = DNSPreprocessor.preprocess_client_dsp_data(file_name)
    df.to_csv(target_file_name, ',', index=False)


def preprocess(data_dir_name: str, out_dir_name: str):
    data_dir = pathlib.Path(data_dir_name)
    out_dir = pathlib.Path(out_dir_name)

    out_dir.mkdir(parents=True, exist_ok=True)

    for file_path in data_dir.glob('dnsSummary_user*.pcap.csv'): #dnsSummary_user292.pcap.csv
        preprocess_file(str(file_path), str(out_dir.joinpath(f'{file_path.name}-proc.csv')))

def shell(data_dir: str):
    ds = data_loader.Dataset.from_dir(data_dir, ['frame_time_relative', 'dns_qry_name', 'dns_qry_type'])
    from IPython import embed; embed()


def run_multiclass(data_dir: str):
    data_loader.load(data_dir, ['frame_time_relative', 'dns_qry_name', 'dns_qry_type'])
    query_name_learner.run_multiclass()

def run(data_dir: str, save_estimator=False, segment_size = query_name_learner.SEGMENT_SIZE, overlap: float = query_name_learner.OVERLAP_FRACTION, is_segment_size_in_seconds: bool = False, ipython_when_done=False):
    data_loader.load(data_dir, ['frame_time_relative', 'dns_qry_name', 'dns_qry_type'])

    query_name_learner.SEGMENT_SIZE = segment_size
    query_name_learner.OVERLAP_FRACTION = overlap
    query_name_learner.IS_SEGMENT_SIZE_IN_SECONDS = is_segment_size_in_seconds

    per_user_states = query_name_learner.run_one_v_all(save_estimator)
    if ipython_when_done:
        from IPython import embed; embed()


def load(file_name: str):
    query_name_learner.load(file_name)


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
