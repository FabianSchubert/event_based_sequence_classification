#!/bin/bash

NJOBS=10

cd ../..

for i in $( seq 1 $NJOBS )
do
    python3 -m experiments.gestures.run
done