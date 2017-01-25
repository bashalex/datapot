#!/bin/bash

in="job.csv.zip"
wget https://www.dropbox.com/s/y2lkws3eyce09v7/job.csv.zip?dl=1 -O $in 
unzip $in
rm $in
in="job.csv"
./prep.py $in
filename=(${in//./ })
out=${filename[0]}$".json"
out2=${filename[0]}$".jsonlines"
cat $out | jq -c -M '.[]' > $out2
truncate -s -1 $out2
rm $out
./job_eval.py
