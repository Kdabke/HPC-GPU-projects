#!/bin/bash
#
#$ -cwd
#$ -pe smp 6
#$ -l mem_free=10G
#
#$ -S /bin/bash
#$ -e /common/bermanblab/data/private_data/POC_ROC/POC_ROC_slide_analysis/json_files_all_51322/scratch
#$ -o /common/bermanblab/data/private_data/POC_ROC/POC_ROC_slide_analysis/json_files_all_51322/scratch

conda activate image_results


python convert_json_immune_count_magn20_62022.py "$@"
