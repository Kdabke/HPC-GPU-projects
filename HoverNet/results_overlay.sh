#!/bin/bash
#
#$ -cwd
#$ -pe smp 6
#$ -l mem_free=50G
#
#$ -S /bin/bash
#$ -e /common/bermanblab/data/private_data/POC_ROC/POC_ROC_slide_analysis/overlay_image_outputs
#$ -o /common/bermanblab/data/private_data/POC_ROC/POC_ROC_slide_analysis/overlay_image_outputs


source /home/dabkek/miniconda3/etc/profile.d/conda.sh
conda activate /home/dabkek/miniconda3/envs/overlay_image


python json_analysis.py "$@"

