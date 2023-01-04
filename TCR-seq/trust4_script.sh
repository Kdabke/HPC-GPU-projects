#!/bin/bash
#
#$ -cwd
#$ -pe smp 6
#$ -l mem_free=20G
#$ -S /bin/bash
#$ -e path_to_working_dir
#$ -o path_to_working_dir

conda activate TRUST4

run-trust4 -b /common/bermanblab/data/private_data/POC_ROC/coetzeesg_pocroc/$1 \
	-f TRUST4-master/hg38_bcrtcr.fa --ref TRUST4-master/human_IMGT+C.fa
