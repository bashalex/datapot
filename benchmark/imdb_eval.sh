#!/bin/bash

in="imdb.tsv"
wget https://www.dropbox.com/s/34wv42gopxnqgbk/labeledTrainData.tsv?dl=1 -O $in 
./benchmark/prep.py $in
filename=(${in//./ })
out=${filename[0]}$".json"
out2=${filename[0]}$".jsonlines"
cat $out | jq -c -M '.[]' > $out2
truncate -s -1 $out2
rm $out
./benchmark/imdb_eval.py
