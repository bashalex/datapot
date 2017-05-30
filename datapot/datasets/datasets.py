from os.path import exists
from os import makedirs
from urllib.request import urlretrieve
import bz2

IMDB_FILENAME = "imdb.jsonlines.bz2"
IMDB_URL = "https://www.dropbox.com/s/5k1nci6m3bqrh4i/imdb.jsonlines.bz2?dl=1"
TINKOFF_FILENAME = "tink.jsonlines.bz2"
TINKOFF_URL = "https://www.dropbox.com/s/gwu8r2zf7ece0ws/tink.jsonlines.bz2?dl=1"
JOB_SALARY_FILENAME = "job.jsonlines.bz2"
JOB_SALARY_URL = "https://www.dropbox.com/s/7clzi7synp07c7d/job.jsonlines.bz2?dl=1"
MALLAT_URL = "https://www.dropbox.com/s/oj64lv620y1h62o/mallat.jsonlines.bz2?dl=1"
MALLAT_FILENAME = "mallat.jsonlines.bz2"
DATA_HOME = 'data/'

def __fetch_dataset(url, filename):
    if not exists(DATA_HOME):
        makedirs(DATA_HOME)
    if not exists(DATA_HOME+filename):
        urlretrieve(url, DATA_HOME+filename)

def __fetch_mallat():
    __fetch_dataset(MALLAT_URL, MALLAT_FILENAME)

def __fetch_imdb():
    __fetch_dataset(IMDB_URL, IMDB_FILENAME)

def __fetch_job_salary():
    __fetch_dataset(JOB_SALARY_URL, JOB_SALARY_FILENAME)

def __fetch_tinkoff():
    __fetch_dataset(TINKOFF_URL, TINKOFF_FILENAME)

def load_mallat(data_home=DATA_HOME):
    """Load and return mallat dataset
    Parameters
    ----------
        data_home : String
            An arbitrary directory location to look in for mallat.jsonlines.bz2

    Returns
    -------
        data : BZ2File
            Bz2-compressed dataset
    """
    if data_home == DATA_HOME:
        __fetch_mallat()
    return bz2.BZ2File(DATA_HOME+MALLAT_FILENAME)

def load_tinkoff(data_home=DATA_HOME):
    """Load and return tinkoff dataset
    Parameters
    ----------
        data_home : String
            An arbitrary directory location to look in for tink.jsonlines.bz2

    Returns
    -------
        data : BZ2File
            Bz2-compressed dataset
    """
    if data_home == DATA_HOME:
        __fetch_tinkoff()
    return bz2.BZ2File(DATA_HOME+TINKOFF_FILENAME)

def load_imdb(data_home=DATA_HOME):
    """Load and return imdb dataset
    Parameters
    ----------
    data_home : String An arbitrary directory location to look in for imdb.jsonlines.bz2
    Returns
    -------
    data : BZ2File
            Bz2-compressed dataset
    """
    if data_home == DATA_HOME:
        __fetch_imdb()
    return bz2.BZ2File(DATA_HOME+IMDB_FILENAME)
def load_job_salary(data_home=DATA_HOME):
    """Load and return job salary dataset
    Parameters
    ----------
        data_home : String
            An arbitrary directory location to look in for job.jsonlines.bz2

    Returns
    -------
        data : BZ2File
            Bz2-compressed dataset
    """
    if data_home == DATA_HOME:
        __fetch_job_salary()
    return bz2.BZ2File(DATA_HOME+JOB_SALARY_FILENAME)


