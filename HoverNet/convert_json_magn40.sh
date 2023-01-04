#!/bin/bash
#
#$ -cwd
#$ -pe smp 6
#$ -l mem_free=10G
#
#$ -S /bin/bash
#$ -e path_to_working_dir
#$ -o path_to_working_dir

conda activate image_results


python convert_json_immune_count_magn40_62022.py "$@"
