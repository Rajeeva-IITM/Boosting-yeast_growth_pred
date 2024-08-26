#!/bin/bash

# cd "/home/rajeeva/Project/boosting/" ||
# Run script

# export seed=1
export suffix="${SEED}_dummy"   # Suffix of the directory created
export extra_params="dummy.run=true dummy.strategy=most_frequent"

python src/tune_model.py data.path=/home/rajeeva/Project/data/chem_subsets/new_latent/carbons/carbons_bloom2013_clf_3_pubchem.feather \
 data.savename="Carbons_Bloom2013_${suffix}" "seed=${SEED}" "${extra_params}"

echo "Carbons_Bloom2013 Done"

python src/tune_model.py data.path=/home/rajeeva/Project/data/chem_subsets/new_latent/carbons/carbons_bloom2015_clf_3_pubchem.feather \
 data.savename="Carbons_Bloom2015_${suffix}" "seed=${SEED}" "${extra_params}"

echo "Carbons_Bloom2015 Done"

python src/tune_model.py data.path=/home/rajeeva/Project/data/chem_subsets/new_latent/carbons/carbons_bloom2019_clf_3_pubchem.feather \
 data.savename="Carbons_Bloom2019_${suffix}" "seed=${SEED}" "${extra_params}"

echo "Carbons_Bloom2019 Done"

python src/tune_model.py data.path=/home/rajeeva/Project/data/chem_subsets/new_latent/carbons/carbons_bloom2019_BYxM22_clf_3_pubchem.feather \
 data.savename="Carbons_Bloom2019_BYxM22_${suffix}" "seed=${SEED}" "${extra_params}"

echo "Carbons_Bloom2019_BYxM22 Done"

python src/tune_model.py data.path=/home/rajeeva/Project/data/chem_subsets/new_latent/carbons/carbons_bloom2019_RMxYPS163_clf_3_pubchem.feather \
 data.savename="Carbons_Bloom2019_RMxYPS163_${suffix}" "seed=${SEED}" "${extra_params}"

echo "Carbons_Bloom2019_RMxYPS163 Done"

echo "All Done"
