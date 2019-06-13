import pickle
import logging
import multiprocessing
from dataclasses import dataclass


import numpy
import joblib

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, OneClassSVM
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

from skorch.net import NeuralNet

import xgboost
from typing import Tuple

import data_loader
import utils

LOG = logging.getLogger(__name__)

SEGMENT_SIZE_IN_SEC = 30*60
MIN_SEGMENT_DNS_QUERIES = 5
OVERLAP_PERCENT=0.2
TRAIN_SET_SIZE = 0.8
TRAIN_TEST_SHUFFLE=True

@dataclass
class State:
    X_train: object
    y_train: object
    X_test: object
    y_test: object

    estimator: BaseEstimator
    y_test_pred: object

    segment_size_in_sec: int = SEGMENT_SIZE_IN_SEC
    min_segment_dns_queries: int = MIN_SEGMENT_DNS_QUERIES



def _split_dns_hostnames_to_segments(df, segment_size_in_sec: int, overlap: float = None):
    segment_start_time = None
    segment = []
    for _, r in df.iterrows():
        if segment_start_time is None:
            segment_start_time = r.frame_time_relative

        if r.frame_time_relative - segment_start_time > segment_size_in_sec: #end of segment
            yield segment
            segment = []
            segment_start_time = r.frame_time_relative

        segment.append(r.dns_qry_name)

    if segment:
        yield segment

def _load_dns_hostname_data_per_user(segment_size_in_sec: int):
    per_user_text_segments = {}
    for user, df in data_loader.dataset:
        LOG.debug(f'Loading data of {user}')
        segments = _split_dns_hostnames_to_segments(df, segment_size_in_sec)
        text_segments = [data_loader.segment_to_text(s) for s in segments]
        per_user_text_segments[user] = text_segments

    return per_user_text_segments



def _save_stuff(stuff, file_name: str):
    joblib.dump(stuff, file_name)

def _load_stuff(file_name: str):
    return joblib.load(file_name)

def _get_estimator_type(estimator):
    estimator = getattr(estimator, 'best_estimator_', estimator)
    estimator = getattr(estimator, '_final_estimator', estimator)
    return estimator.__class__.__name__

def _safe_get_best_estimator(estimator):
    return getattr(estimator, 'best_estimator_', estimator)

def _split_train_test(per_user_segments: dict, train_size=TRAIN_SET_SIZE, shuffle=False) -> Tuple[dict, dict]:
    per_user_train = {}
    per_user_test = {}
    for user, segments in per_user_segments.items():
        per_user_train[user], per_user_test[user] = train_test_split(segments, shuffle=shuffle, train_size=train_size)

    return per_user_train, per_user_test

def _per_user_segments_to_X_and_y(per_user_segments):
    all_users = sorted(list(per_user_segments.keys()))
    yenc = LabelEncoder()
    yenc.fit(all_users)

    X = []
    y_parts = []
    y = []
    for user, segments in per_user_segments.items():
        X += segments
        y_parts.append( yenc.transform([user]).repeat(len(segments)) )

    y = numpy.concatenate(y_parts)
    return X, y

def run_multiclass(out_file_name=None):
    segments_file_name = f'segments_per_user_{SEGMENT_SIZE_IN_SEC}_{MIN_SEGMENT_DNS_QUERIES}.job.xz'

    try:
        per_user_segments = _load_stuff(segments_file_name)
    except FileNotFoundError:
        per_user_segments = _load_dns_hostname_data_per_user(segment_size_in_sec=SEGMENT_SIZE_IN_SEC)
        _save_stuff(per_user_segments, segments_file_name)

    per_user_train, per_user_test = _split_train_test(per_user_segments, shuffle=TRAIN_TEST_SHUFFLE, train_size=TRAIN_SET_SIZE)

    X_train, y_train = _per_user_segments_to_X_and_y(per_user_train)
    X_test, y_test = _per_user_segments_to_X_and_y(per_user_test)

    param_grid = {
        # 'svc__C': [1, 5],
        # 'mnb__alpha' : [1e-10, 1e-5, 0.1, 0.5],

        'feature__ngram_range': [(1, 4)],
        'mnb__alpha': [1e-3],
        # 'cnb__alpha': [1e-10, 1e-5, 0.1, 0.5],

        # 'xgb__max_depth': [5]
    }

    pipeline = Pipeline(
        steps=[
            ('feature', TfidfVectorizer()),
            # ('svc', SVC()),
            ('mnb', MultinomialNB())
            # ('cnb', ComplementNB()),
            # ('xgb', xgboost.XGBClassifier(max_depth=5)),
            # ('ann', MLPClassifier((1000,200, 100,))),
        ]
    )
    estimator = GridSearchCV(pipeline, param_grid, iid=True, cv=5, verbose=3, n_jobs=multiprocessing.cpu_count()-1)
    # estimator = pipeline
    estimator.fit(X_train, y_train)

    best_score = getattr(estimator, 'best_score_', None)
    best_params = getattr(estimator, 'best_params_', None)
    if best_score:
        LOG.info(f'Best score: {best_score} | param: {best_params}')

    y_test_pred = estimator.predict(X_test)
    LOG.info(f'Test report:\n{metrics.classification_report(y_test, y_test_pred)}')

    if out_file_name is None:
        score = metrics.f1_score(y_test, y_test_pred, average='weighted')
        out_file_name = f'{_get_estimator_type(estimator)}-{score:.2f}-{SEGMENT_SIZE_IN_SEC}-{MIN_SEGMENT_DNS_QUERIES}-{"TRAIN_TEST_SHUFFLE" if TRAIN_TEST_SHUFFLE else "NO_TRAIN_TEST_SHUFFLE"}-{utils.get_current_time_stamp()}.job.gz'

    LOG.info( f'Saving state to: {out_file_name}')
    _save_stuff(State(X_train, X_test, y_train, y_test, estimator, y_test_pred), out_file_name)
    LOG.debug( 'Done')

    from IPython import embed;embed()

def load(file_name: str):
    state: State = _load_stuff(file_name)
    from IPython import embed; embed()
