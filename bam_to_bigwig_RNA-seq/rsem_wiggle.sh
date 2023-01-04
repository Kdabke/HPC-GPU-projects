#!/bin/bash
#
#$ -cwd
#$ -pe smp 6
#$ -l mem_free=10G
#
#$ -S /bin/bash
#$ -e /common/bermanblab/data/private_data/POC_ROC/POC_ROC_RNA-Seq/POC_ROC_phase1_bw/scratch
#$ -o /common/bermanblab/data/private_data/POC_ROC/POC_ROC_RNA-Seq/POC_ROC_phase1_bw/scratch

set -xe

INFILE="/common/bermanblab/data/private_data/POC_ROC/coetzeesg_pocroc/phase1/rsem_trimmed_repair/$1.genome.sorted.bam"
OUTFILE="$1.wig"
NAME="$1"

[[ -e "${INFILE}" ]] && echo "infile ${INFILE} exists"

source /home/dabkek/miniconda3/etc/profile.d/conda.sh
conda activate /home/dabkek/miniconda3/envs/rnaseq

rsem-bam2wig \
	${INFILE} \
	${OUTFILE} \
	${NAME}

