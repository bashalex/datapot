#!/usr/bin/env python
import pandas as pd
import sys

filename = sys.argv[1]
new_filename = filename.split('.')[0] + '.json'
data = pd.read_csv(filename, delimiter='\t')
data.to_json(path_or_buf=new_filename, orient='records')
