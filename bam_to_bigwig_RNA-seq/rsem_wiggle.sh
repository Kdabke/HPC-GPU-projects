#!/bin/bash
#
#$ -cwd
#$ -pe smp 6
#$ -l mem_free=10G
#
#$ -S /bin/bash
#$ -e path_to_working_dir
#$ -o path_to_working_dir

set -xe

INFILE="/common/bermanblab/data/private_data/POC_ROC/coetzeesg_pocroc/phase1/rsem_trimmed_repair/$1.genome.sorted.bam"
OUTFILE="$1.wig"
NAME="$1"

[[ -e "${INFILE}" ]] && echo "infile ${INFILE} exists"

conda activate rnaseq

rsem-bam2wig \
	${INFILE} \
	${OUTFILE} \
	${NAME}

