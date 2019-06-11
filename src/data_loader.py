import logging
import re

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
import pathlib2

LOG = logging.getLogger(__name__)

DATA_FILE_NAME_PATTERN = re.compile(r'dnsSummary_(.*)\.pcap\.csv-proc\.csv') #dnsSummary_user292.pcap.csv-proc.csv

def normalize_domain_name(domain_name):
    return domain_name.replace('.', '__')

def segment_to_text(s):
    return '\n'.join(normalize_domain_name(x) for x in s)


class Dataset(object):
    @classmethod
    def from_dir(cls, data_dir: str, usecols=None):
        users_and_dfs = []

        dir_path = pathlib2.Path(data_dir)
        for file_path in dir_path.glob('*proc.csv'):
            mo = DATA_FILE_NAME_PATTERN.match(file_path.name)
            if mo is None:
                LOG.warning(f'{file_path} does not match expected pattern')
                continue

            username = mo.group(1)
            df = pandas.read_csv(file_path, usecols=usecols)

            users_and_dfs.append((username, df))

        return cls(users_and_dfs)

    def __init__(self, users_and_dfs):
        self.users_and_dfs = users_and_dfs


        self.tfidf = TfidfVectorizer(use_idf=True)
        domains_texts = []
        for _, df in self.users_and_dfs:
            domains_texts.append(segment_to_text(df['dns_qry_name'].unique()))
        self.tfidf.fit(domains_texts)

    def __iter__(self):
        return iter(self.users_and_dfs)

dataset: Dataset = None

def load(data_dir: str, usecols=None):
    global dataset
    if dataset is not None:
        raise RuntimeError('Dataset already loaded')

    dataset = Dataset.from_dir(data_dir, usecols)
