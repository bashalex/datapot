import bz2
import csv
import io
import json


def create_full_iterator(data):
    if isinstance(data, (list, tuple)):
        return identity_iterator

    if hasattr(data, 'name'):
        filename = data.name
    elif isinstance(data, bz2.BZ2File):
        filename = data._fp.name
    else:
        raise ValueError('This format is not supported')

    filename_parts = filename.split('.')
    if len(filename_parts) < 2:
        return jsonlines_file_iterator
    if 'jsonlines' in filename_parts[-2:]:
        return jsonlines_file_iterator
    if filename_parts[-1] == 'csv':
        return csv_file_iterator


def _move_pointer_to_start(data):
    try:
        file_types = (file, io.IOBase, bz2.BZ2File)
    except NameError:
        file_types = (io.IOBase, bz2.BZ2File,)
    if isinstance(data, file_types):
        data.seek(0, 0)


def jsonlines_file_iterator(data):
    decoder = json.JSONDecoder()
    _move_pointer_to_start(data)

    for obj in data:
        if isinstance(obj, bytes):
            obj = obj.decode('utf-8')
        obj = decoder.decode(obj)
        yield obj

def csv_file_iterator(data):
    _move_pointer_to_start(data)
    reader = csv.DictReader(data)
    for obj in reader:
        yield obj

def identity_iterator(data):
   for obj in data:
       yield obj
