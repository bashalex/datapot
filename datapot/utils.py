import re
import json
import pandas as pd


def csv_to_jsonlines(csv_file, jsonlines_file, **pd_read_csv_options):
    with open(csv_file, 'r') if isinstance(csv_file, str) else csv_file as csv_file, \
         open(jsonlines_file, 'w') if isinstance(jsonlines_file, str) else jsonlines_file as jsonlines_file:
        df = pd.read_csv(csv_file, **pd_read_csv_options)
        json_string = df.to_json(orient='records')[1:-1]
        jsonlines_string = re.sub('},', '}\n', json_string)
        jsonlines_file.write(jsonlines_string)


def json_to_jsonlines(json_file_path, output_file_path='json_lines.jsonl'):
    '''
    Creat JSON Lines file from JSON
    :param json_file_path: path to JSON file
    :param output_file_path: output file path
    '''
    with open(json_file_path) as ftr:
        json_file = json.load(ftr)

    with open(output_file_path, 'w') as outfile:
        for entry in json_file:
            json.dump(entry, outfile)
            outfile.write('\n')
