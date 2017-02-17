import re

import pandas as pd

def csv_to_jsonlines(input_file, output_file, **pd_read_csv_options):
    close_input_file = close_output_file = False
    if isinstance(input_file, str):
        input_file = open(input_file, 'r')
        close_input_file = True
    if isinstance(output_file, str):
        output_file = open(output_file, 'w')
        close_output_file = True

    df = pd.read_csv(input_file, **pd_read_csv_options)
    json_string = df.to_json(orient='records')[1:-1]
    jsonlines_string = re.sub('},', '}\n', json_string)
    output_file.write(jsonlines_string)
    if close_input_file:
        input_file.close()
    if close_output_file:
        output_file.close()