import json
import sys

with open(str(sys.argv[1])) as ftr:
    json_list = json.load(ftr)

with open(str(sys.argv[2]), "w") as ftw:
    for obj in json_list:
        print(json.dumps(obj), file=ftw)
