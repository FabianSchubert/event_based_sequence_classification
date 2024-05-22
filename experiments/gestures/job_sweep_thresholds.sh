#!/bin/bash

N_TRIALS=10

cd ../..

N_SWEEP_TH=10

TH_MIN=0.01
TH_MAX=10.0

TH_LOG_MIN=$(echo "l($TH_MIN)/l(10)" | bc -l)
TH_LOG_MAX=$(echo "l($TH_MAX)/l(10)" | bc -l)

TH_SWEEP=$(seq $TH_LOG_MIN $(echo "($TH_LOG_MAX - $TH_LOG_MIN) / ($N_SWEEP_TH - 1)" | bc -l) $TH_LOG_MAX | awk '{ print 10^$1 }')

echo $TH_SWEEP

for TH in $TH_SWEEP
do
    for i in $( seq 1 $N_TRIALS )
    do
        python3 -m experiments.gestures.run --threshold_scale $TH
    done
done