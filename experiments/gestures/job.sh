#!/bin/bash

NJOBS=10

python3 settings.py

cd ../..

for i in $( seq 1 $NJOBS )
do
    python3 -m experiments.gestures.run
done