#!/bin/bash

# cd "/home/rajeeva/Project/boosting/" || exit

for seed in {1..5}
do
    export SEED=${seed}
    source ./scripts/run_script.sh
done
