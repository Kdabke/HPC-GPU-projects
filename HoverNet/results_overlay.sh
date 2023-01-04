#!/bin/bash
#
#$ -cwd
#$ -pe smp 6
#$ -l mem_free=50G
#
#$ -S /bin/bash
#$ -e path_to_working_dir
#$ -o path_to_working_dir



conda activate overlay_image


python json_analysis.py "$@"

