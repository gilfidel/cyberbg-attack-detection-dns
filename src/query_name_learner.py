import os
import pickle
import logging
import multiprocessing
from dataclasses import dataclass


import numpy
import joblib

import pandas
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, OneClassSVM
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import humanize

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

import xgboost

from typing import Tuple

import data_loader
import utils

LOG = logging.getLogger(__name__)

SEGMENT_SIZE = 50
OVERLAP_FRACTION=0.2
IS_SEGMENT_SIZE_IN_SECONDS = False
MIN_SEGMENT_SIZE = 5

TRAIN_SET_FRACTION = 0.8
TEST_SET_FRACTION = 0.2

@dataclass
class State:
    X_train: object
    y_train: object
    X_test: object
    y_test: object

    estimator: BaseEstimator
    y_test_pred: object
    y_test_proba: object = None

    segment_size: int = SEGMENT_SIZE
    overlap_fraction: int = OVERLAP_FRACTION


def _split_dns_hostnames_to_segments_by_time(df, segment_size_in_sec: int, overlap_fraction: float):
    overlap_size = segment_size_in_sec * overlap_fraction
    step_size_in_sec = segment_size_in_sec - overlap_size

    df_start_time = df.iloc[0].frame_time_relative
    df_end_time = df.iloc[-1].frame_time_relative

    for seg_start_time in range(int(df_start_time), int(df_end_time - segment_size_in_sec), int(step_size_in_sec)):
        segment = df[numpy.logical_and(
                    df.frame_time_relative >= seg_start_time,
                    df.frame_time_relative <= seg_start_time + segment_size_in_sec)]['dns_qry_name'].values

        if len(segment) > MIN_SEGMENT_SIZE:
            yield segment

# def _split_dns_hostnames_to_segments_by_time(df, segment_size_in_sec: int, overlap_fraction: float):
#     segment_start_time = None
#     segment = []
#     for _, r in df.iterrows():
#         if segment_start_time is None:
#             segment_start_time = r.frame_time_relative
#
#         if r.frame_time_relative - segment_start_time > segment_size_in_sec: #end of segment
#             yield segment
#             segment = []
#             segment_start_time = r.frame_time_relative
#
#         segment.append(r.dns_qry_name)
#
#     if segment:
#         yield segment

def _split_dns_hostname_to_segments_by_count(df, segment_size: int, overlap_fraction: float):
    overlap_size = int(segment_size * overlap_fraction)
    segment_step_size = segment_size - overlap_size

    for idx in range(0, len(df) - segment_size, segment_step_size):
        segment = df[idx:idx+segment_step_size]['dns_qry_name'].values
        yield segment

def _split_train_test_by_time(df: pandas.DataFrame, train_set_fraction: float, test_set_fraction: float):
    total_time_span = df.iloc[-1].frame_time_relative - df.iloc[0].frame_time_relative
    train_df = df[df.frame_time_relative <= train_set_fraction * total_time_span]
    test_df = df[df.frame_time_relative > total_time_span - test_set_fraction * total_time_span]

    return train_df, test_df

def _split_train_test_by_count(df: pandas.DataFrame, train_set_fraction: float, test_set_fraction: float):
    train_df = df[:int(len(df) * train_set_fraction)]
    test_df = df[int(len(df) * (1 - test_set_fraction)):]

    return train_df, test_df

def _load_dns_hostname_data_per_user():
    per_user_train_and_test = {}

    segment_split_func = _split_dns_hostnames_to_segments_by_time if IS_SEGMENT_SIZE_IN_SECONDS else _split_dns_hostname_to_segments_by_count
    train_set_split_func = _split_train_test_by_time if IS_SEGMENT_SIZE_IN_SECONDS else _split_train_test_by_count

    for user, df in data_loader.dataset:
        LOG.debug(f'Loading data of {user}')

        train_df, test_df = train_set_split_func(df, TRAIN_SET_FRACTION, TEST_SET_FRACTION)

        train_segments = [data_loader.segment_to_text(s) for s in segment_split_func(train_df, SEGMENT_SIZE, OVERLAP_FRACTION)]
        test_segments = [data_loader.segment_to_text(s) for s in segment_split_func(test_df, SEGMENT_SIZE, OVERLAP_FRACTION)]

        LOG.debug(f'{user}:: Train set: {humanize.naturalsize(len(train_df))} rows / {humanize.naturalsize(len(train_segments))} segments | Test set: {humanize.naturalsize(len(test_df))} rows / {humanize.naturalsize(len(test_segments))} segments')

        per_user_train_and_test[user] = (train_segments, test_segments)


    return per_user_train_and_test

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


def _per_user_segments_to_multiclass_X_and_y(per_user_segments):
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

def _per_user_segments_to_per_user_one_v_all_X_and_y(per_user_segments):
    per_user_X_and_y = {}

    for user in per_user_segments.keys():
        user_segments = per_user_segments[user]
        other_segments = sum([segs for other_user, segs in per_user_segments.items() if other_user != user], [])

        X = user_segments + other_segments
        y = numpy.concatenate(
            (
                numpy.array([1]).repeat(len(user_segments)),
                numpy.array([0]).repeat(len(other_segments)),
            ),
        )

        per_user_X_and_y[user] = X,y

    return per_user_X_and_y

def _create_grid_searcher():
    param_grid = {
        # 'svc__C': [1, 5],
        # 'mnb__alpha' : [1e-10, 1e-5, 0.1, 0.5],

        'feature__ngram_range': [(1,3),(1, 4), (1,5), (1,6)],
        'nb__alpha': [1e-10, 1e-3, 1e-5],
        # 'xgb__max_depth': [5]
    }

    pipeline = _create_estimator()
    estimator = GridSearchCV(pipeline, param_grid, iid=True, cv=5, verbose=3, n_jobs=multiprocessing.cpu_count() - 1, scoring='f1')

    return estimator


def _create_keras_model(optimizer=tf.train.AdamOptimizer(1e-3), init='glorot_uniform'):
    LOG.info('_create_keras_model')
    try:
        device_name = os.environ['COLAB_TPU_ADDR']
        TPU_ADDRESS = 'grpc://' + device_name
    except KeyError:
        TPU_ADDRESS = None

    model = Sequential(
        layers=[
            Dense(1000, input_shape=(2000,),kernel_initializer=init, activation='relu'),
            Dense(200, kernel_initializer=init, activation='relu'),
            Dense(200, kernel_initializer=init, activation='relu'),
            Dense(200, kernel_initializer=init, activation='relu'),
            Dense(10, kernel_initializer=init, activation='relu'),
            Dense(1, kernel_initializer=init, activation='sigmoid')
        ]
    )
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'] )
    if TPU_ADDRESS:
        LOG.info(f'Converting model to TPU @ {TPU_ADDRESS}')
        model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)))
    return model


def _create_estimator():
    pipeline = Pipeline(
        steps=[
            ('feature', TfidfVectorizer(ngram_range=(1,4))),
            # ('svc', SVC()),
            ('nb', MultinomialNB(alpha=1e-3))
            # ('cnb', ComplementNB(alpha=1e-3)),
            # ('xgb', xgboost.XGBClassifier(max_depth=5)),
            #('ann', MLPClassifier((200, 10), verbose=True)),
            # ('keras', KerasClassifier(build_fn=_create_keras_model, ))
        ]
    )

    return pipeline

def _segments_to_X_and_y(user_segments, other_segments):
    X = user_segments + other_segments
    y = numpy.concatenate(
        (
            numpy.array([1]).repeat(len(user_segments)),
            numpy.array([0]).repeat(len(other_segments)),
        ),
    )
    return X,y

def _per_user_segments_to_1vAll_train_and_test_X_and_y(per_user_segments):
    per_user_X_and_y = {}

    for user in per_user_segments.keys():
        user_train_segments, user_test_segments = per_user_segments[user]
        other_train_segments = sum([train_segs for other_user, (train_segs,_) in per_user_segments.items() if other_user != user], [])
        other_test_segments = sum([test_segs for other_user, (_, test_segs) in per_user_segments.items() if other_user != user], [])

        X_train, y_train = _segments_to_X_and_y(user_train_segments, other_train_segments)
        X_test, y_test = _segments_to_X_and_y(user_test_segments, other_test_segments)

        per_user_X_and_y[user] = ((X_train, y_train), (X_test, y_test))

    return per_user_X_and_y

def run_one_v_all(save_estimator=True):
    LOG.info(
        f'run_one_v_all: SEGMENT_SIZE: {SEGMENT_SIZE} | OVERLAP: {OVERLAP_FRACTION:.2f} save_estimator: {save_estimator}')

    per_user_segments = _load_per_user_train_and_test()


    per_user_train_and_test_X_and_y = _per_user_segments_to_1vAll_train_and_test_X_and_y(per_user_segments)

    per_user_states = {}
    ap_scores = []
    f1_scores = []
    for user, ((X_train, y_train), (X_test, y_test)) in per_user_train_and_test_X_and_y.items():
        LOG.info( f'{user}:: Training')
        estimator = _create_grid_searcher()
        # estimator = _create_estimator()


        estimator.fit(X_train, y_train)

        best_score = getattr(estimator, 'best_score_', None)
        best_params = getattr(estimator, 'best_params_', None)
        if best_score:
            LOG.info(f'Best score: {best_score} | param: {best_params}')

        y_test_pred = estimator.predict(X_test)
        y_test_proba = estimator.predict_proba(X_test)

        ap_score = metrics.average_precision_score(y_test, y_test_proba[:,1])
        f1_score = metrics.f1_score(y_test, y_test_pred)
        LOG.info(f'{user}:: test set: average_precision score={ap_score:.3f}, f1_score = {f1_score:.3f}')

        ap_scores.append(ap_score)
        f1_scores.append(f1_score)

        per_user_states[user] = State(X_train, y_train, X_test, y_test, estimator if save_estimator else None, y_test_pred, y_test_proba)


    out_file_name = f'1vAll-_{TRAIN_SET_FRACTION:.2f}_{TEST_SET_FRACTION:.2f}_{_get_estimator_type(estimator)}-f1_{numpy.mean(f1_scores):.2f}-ap_{numpy.mean(ap_scores):.3f}-{"TIME" if IS_SEGMENT_SIZE_IN_SECONDS else "COUNT"}-{SEGMENT_SIZE}-{OVERLAP_FRACTION:.2f}-{utils.get_current_time_stamp()}.job.gz'
    LOG.info(f'Saving state to: {out_file_name}')
    _save_stuff(per_user_states, out_file_name)
    LOG.debug('Done')

    return per_user_states

# def run_multiclass(out_file_name=None):
#     per_user_train, per_user_test = _load_per_user_train_and_test()
#
#     X_train, y_train = _per_user_segments_to_multiclass_X_and_y(per_user_train)
#     X_test, y_test = _per_user_segments_to_multiclass_X_and_y(per_user_test)
#
#
#     estimator = _create_grid_searcher()
#     estimator.fit(X_train, y_train)
#
#
#     best_score = getattr(estimator, 'best_score_', None)
#     best_params = getattr(estimator, 'best_params_', None)
#     if best_score:
#         LOG.info(f'Best score: {best_score} | param: {best_params}')
#
#     y_test_pred = estimator.predict(X_test)
#     LOG.info(f'Test report:\n{metrics.classification_report(y_test, y_test_pred)}')
#
#     from IPython import embed; embed()
#
#     if out_file_name is None:
#         score = metrics.f1_score(y_test, y_test_pred, average='weighted')
#         out_file_name = f'multi-{_get_estimator_type(estimator)}-{score:.2f}-{SEGMENT_SIZE_IN_SEC}-{MIN_SEGMENT_DNS_QUERIES}-{"TRAIN_TEST_SHUFFLE" if TRAIN_TEST_SHUFFLE else "NO_TRAIN_TEST_SHUFFLE"}-{utils.get_current_time_stamp()}.job.gz'
#
#     LOG.info( f'Saving state to: {out_file_name}')
#     _save_stuff(State(X_train, y_train, X_test, y_test, estimator, y_test_pred), out_file_name)
#     LOG.debug( 'Done')
#
#     from IPython import embed;embed()


def _load_per_user_train_and_test():
    segments_file_name = f'segments_per_user_{TRAIN_SET_FRACTION:.2f}_{TEST_SET_FRACTION:.2f}_{"TIME" if IS_SEGMENT_SIZE_IN_SECONDS else "COUNT"}_{SEGMENT_SIZE}_{OVERLAP_FRACTION:.2f}.job.xz'
    try:
        per_user_segments = _load_stuff(segments_file_name)
    except FileNotFoundError:
        per_user_segments = _load_dns_hostname_data_per_user()
        _save_stuff(per_user_segments, segments_file_name)

    return per_user_segments


def load(file_name: str):
    x = _load_stuff(file_name)
    from IPython import embed; embed()
