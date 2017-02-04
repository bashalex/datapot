from urllib.request import urlretrieve
from os.path import exists
from os import makedirs

IMDB_FILENAME = "imdb.jsonlines.bz2"
IMDB_URL = "https://www.dropbox.com/s/5k1nci6m3bqrh4i/imdb.jsonlines.bz2?dl=1"
TINKOFF_FILENAME = "tink.jsonlines.bz2"
TINKOFF_URL = "https://www.dropbox.com/s/gwu8r2zf7ece0ws/tink.jsonlines.bz2?dl=1"
JOB_SALARY_FILENAME = "job.jsolines.bz2"
JOB_SALARY_URL = "https://www.dropbox.com/s/7v1m2lfhhx92tcm/job.jsonlines.bz2?dl=1"

DATA_HOME = 'data/'

def __fetch_dataset(url, filename):
    if not exists(DATA_HOME):
        makedirs(DATA_HOME)
    if not exists(DATA_HOME+filename):
        urlretrieve(url, DATA_HOME+filename)

def fetch_imdb():
    __fetch_dataset(IMDB_URL, IMDB_FILENAME)

def fetch_job_salary():
    __fetch_dataset(JOB_SALARY_URL, JOB_SALARY_FILENAME)

def fetch_tinkoff():
    __fetch_dataset(TINKOFF_URL, TINKOFF_FILENAME)
