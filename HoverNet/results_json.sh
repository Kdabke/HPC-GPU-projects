#!/bin/bash
#
#$ -cwd
#$ -pe smp 6
#$ -l mem_free=10G
#
#$ -S /bin/bash
#$ -e /common/bermanblab/data/private_data/POC_ROC/POC_ROC_slide_analysis/json_files_all_51322/scratch
#$ -o /common/bermanblab/data/private_data/POC_ROC/POC_ROC_slide_analysis/json_files_all_51322/scratch

source /home/dabkek/miniconda3/etc/profile.d/conda.sh
conda activate /home/dabkek/miniconda3/envs/image_results


python json_analysis_20_40.py "$@"
