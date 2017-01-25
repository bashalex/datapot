#!/bin/bash

in="tink.csv"
wget https://www.dropbox.com/s/w892ypctlymp26m/tink.csv?dl=1 -O $in 
./prep.py $in
filename=(${in//./ })
out=${filename[0]}$".json"
out2=${filename[0]}$".jsonlines"
cat $out | jq -c -M '.[]' > $out2
truncate -s -1 $out2
rm $out
./tink_eval.py
