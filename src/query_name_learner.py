import pickle
import logging
import multiprocessing
import numpy
import joblib

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

import data_loader
import utils

LOG = logging.getLogger(__name__)

def _split_dns_hostnames_to_segments(df, segment_size_in_sec: float):
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

def _preprocess_dns_hostname_data_multiclass(segment_size_in_sec: float):
    X = []
    users = []
    unique_users = []
    for user, df in data_loader.dataset:
        LOG.debug(f'Loading data of {user}')
        segments = _split_dns_hostnames_to_segments(df, segment_size_in_sec)
        text_segments =  [data_loader.segment_to_text(s) for s in segments]
        X += text_segments
        users += [user]* len(text_segments)

        unique_users.append(user)

    yenc = LabelEncoder()
    yenc.fit(unique_users)
    y = yenc.transform(users)

    return X,y

def _preprocess_dns_hostname_data_per_user(segment_size_in_sec: float):
    per_user_text_segments = {}
    for user, df in data_loader.dataset:
        LOG.debug(f'Loading data of {user}')
        segments = _split_dns_hostnames_to_segments(df, segment_size_in_sec)
        text_segments = [data_loader.segment_to_text(s) for s in segments]
        per_user_text_segments[user] = text_segments

    return per_user_text_segments

SEGMENT_SIZE_IN_SEC = 30*60

def _save_stuff(stuff, file_name: str):
    joblib.dump(stuff, file_name)

def _load_stuff(file_name: str):
    return joblib.load(file_name)

TRAIN_SET_SIZE = 0.8

def _get_estimator_type(estimator):
    estimator = getattr(estimator, 'best_estimator_', estimator)
    estimator = getattr(estimator, '_final_estimator', estimator)
    return estimator.__class__.__name__

def _safe_get_best_estimator(estimator):
    return getattr(estimator, 'best_estimator_', estimator)

def run_multiclass(out_file_name=None):
    segments_file_name = f'segments_{SEGMENT_SIZE_IN_SEC}.pickle'

    try:
        X_segments, y_segments = _load_stuff(segments_file_name)
    except FileNotFoundError:
        X_segments, y_segments = _preprocess_dns_hostname_data_multiclass(segment_size_in_sec=SEGMENT_SIZE_IN_SEC)
        _save_stuff((X_segments, y_segments), segments_file_name)

    X_train, X_test, y_train, y_test = train_test_split(X_segments, y_segments, stratify=y_segments, train_size=TRAIN_SET_SIZE)

    param_grid = {
        # 'svc__C': [1, 5],
        # 'feature__ngram_range' : [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7)],
        # 'mnb__alpha' : [1e-10, 1e-5, 0.1, 0.5],

        'feature__ngram_range': [(1,1),(1,2),(1,3),(1, 4)],
        'mnb__alpha': [1e-10, 1e-5, 0.1, 0.5],
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
        out_file_name = f'{_get_estimator_type(estimator)}-{score:.2f}-{utils.get_current_time_stamp()}.job'

    LOG.info( f'Saving state to: {out_file_name}')
    _save_stuff((estimator, X_train, X_test, y_train, y_test, y_test_pred), out_file_name)

    from IPython import embed;embed()

def load(file_name: str):
    estimator, X_train, X_test, y_train, y_test, y_test_pred = _load_stuff(file_name)
    from IPython import embed; embed()

def run_oneclass():
    segments_file_name = f'segments_{SEGMENT_SIZE_IN_SEC}-per_user.pickle'
    try:
        per_user_text_segments = _load_stuff(segments_file_name)
    except FileNotFoundError:
        per_user_text_segments = _preprocess_dns_hostname_data_per_user(segment_size_in_sec=SEGMENT_SIZE_IN_SEC)
        _save_stuff(per_user_text_segments, segments_file_name)

    for user, text_segments in per_user_text_segments.items():
        LOG.info(f'Training on {user} data')

        train_text_segments, test_text_segments = train_test_split(text_segments, train_size=0.8 )

        param_grid = {
            # 'svc__C': [1, 5],
            # 'svm_': [1e-10, 1.0]
        }

        pipeline = Pipeline(
            steps=[
                ('feature', TfidfVectorizer()),
                # ('svc', SVC(decision_function_shape='ovr'))
                ('svm', OneClassSVM())
            ]
        )

        pipeline.fit(train_text_segments)

        from IPython import embed; embed()

        grid = GridSearchCV(pipeline, param_grid, iid=True, cv=5, verbose=3, n_jobs=multiprocessing.cpu_count() - 1)
        LOG.info(f'Best score: {grid.best_score_} | param: {grid.best_params_}')
        from IPython import embed;
        embed()
