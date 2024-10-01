#!/bin/bash

# cd "/home/rajeeva/Project/boosting/" ||
# Run script

# export seed=1
export SEED="1,2,3,4,5"   # Suffix of the directory created
export extra_params='data.savedir=${oc.env:RUN_DIR}/classification/0.25_sigma/${data.savename}_${seed}_${run_type} n_trials=50'

python src/tune_model.py --multirun 'data.path=${oc.env:DATA_DIR}/full/varying_sigma/sigma_0.25/bloom2013_clf.feather' \
 data.savename="Full_Bloom2013" "seed=${SEED}" $extra_params

echo "Carbons_Bloom2013 Done"

python src/tune_model.py --multirun 'data.path=${oc.env:DATA_DIR}/full/varying_sigma/sigma_0.25/bloom2015_clf.feather' \
 data.savename="Carbons_Bloom2015" "seed=${SEED}" $extra_params

echo "Carbons_Bloom2015 Done"

python src/tune_model.py --multirun 'data.path=${oc.env:DATA_DIR}/full/varying_sigma/sigma_0.25/bloom2019_clf.feather' \
 data.savename="Carbons_Bloom2019_BYxRM" "seed=${SEED}" $extra_params

echo "Carbons_Bloom2019 Done"

python src/tune_model.py --multirun 'data.path=${oc.env:DATA_DIR}/full/varying_sigma/sigma_0.25/bloom2019_BYxM22_clf.feather' \
 data.savename="Carbons_Bloom2019_BYxM22" "seed=${SEED}" $extra_params

echo "Carbons_Bloom2019_BYxM22 Done"

python src/tune_model.py --multirun 'data.path=${oc.env:DATA_DIR}/full/varying_sigma/sigma_0.25/bloom2019_RMxYPS163_clf.feather' \
 data.savename="Carbons_Bloom2019_RMxYPS163" "seed=${SEED}" $extra_params

echo "Carbons_Bloom2019_RMxYPS163 Done"

echo "All Done"
