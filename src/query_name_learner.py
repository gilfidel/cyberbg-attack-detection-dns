import pickle
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

import data_loader

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

def _preprocess_dns_hostname_data(segment_size_in_sec: float):
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

SEGMENT_SIZE_IN_SEC = 30*60

def _save_stuff(stuff, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(stuff, f, pickle.HIGHEST_PROTOCOL)

def _load_stuff(file_name: str):
    with open(file_name, 'rb') as f:
        stuff = pickle.load(f)

    return stuff

def run():
    segments_file_name = f'segments_{SEGMENT_SIZE_IN_SEC}.pickle'
    try:
        X_segments, y_segments = _load_stuff(segments_file_name)
    except FileNotFoundError:
        X_segments, y_segments = _preprocess_dns_hostname_data(segment_size_in_sec=SEGMENT_SIZE_IN_SEC)
        _save_stuff((X_segments, y_segments), segments_file_name)

    param_grid = {
        # 'svc__C': [1, 5],
        'mnb__alpha' : [0.0, 0.1, 1, 0.5]
    }

    pipeline = Pipeline(
        steps=[
            ('feature', data_loader.dataset.tfidf),
            # ('svc', SVC(decision_function_shape='ovr'))
            ('mnb', MultinomialNB())
        ]
    )
    grid = GridSearchCV(pipeline, param_grid, iid=True, cv=5, verbose=1)
    grid.fit(X_segments, y_segments)
    from IPython import embed; embed()
