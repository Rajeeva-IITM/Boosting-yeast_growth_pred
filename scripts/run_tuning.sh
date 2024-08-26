#!/bin/bash

# cd /home/rajeeva/Project/boosting/

export train_data=/home/rajeeva/Project/data/chem_subsets/new_latent/carbons/carbons_bloom2019_clf_3_pubchem.feather
export train_name=Bloom2019
export suffix="robust_samples"


# python data_sampling_analysis.py data.train_data=${train_data} \
#  data.savename="Carbons_${train_name}_on_Bloom2013_${suffix}" \
#  data.test_data=/home/rajeeva/Project/data/chem_subsets/new_latent/carbons/carbons_bloom2013_clf_3_pubchem.feather
# echo "${train_name} on Bloom2013"

python data_sampling_analysis.py data.train_data=${train_data} \
 data.savename="Carbons_${train_name}_on_Bloom2015_${suffix}" \
 data.test_data=/home/rajeeva/Project/data/chem_subsets/new_latent/carbons/carbons_bloom2015_clf_3_pubchem.feather
echo "${train_name} on Bloom2013"

python data_sampling_analysis.py data.train_data=${train_data} \
 data.savename="Carbons_${train_name}_on_Bloom2019_BYxRM_${suffix}" \
 data.test_data=/home/rajeeva/Project/data/chem_subsets/new_latent/carbons/carbons_bloom2019_clf_3_pubchem.feather
echo "${train_name} on Bloom2019_new"

python data_sampling_analysis.py data.train_data=${train_data} \
 data.savename="Carbons_${train_name}_on_Bloom2019_BYxM22_${suffix}" \
 data.test_data=/home/rajeeva/Project/data/chem_subsets/new_latent/carbons/carbons_bloom2019_BYxM22_clf_3_pubchem.feather
echo "${train_name} on Bloom2019_BYxM22"

python data_sampling_analysis.py data.train_data=${train_data} \
 data.savename="Carbons_${train_name}_on_Bloom2019_RMxYPS163_${suffix}" \
 data.test_data=/home/rajeeva/Project/data/chem_subsets/new_latent/carbons/carbons_bloom2019_RMxYPS163_clf_3_pubchem.feather
echo "${train_name} on Bloom2019_RMxYPS163"
