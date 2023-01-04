#!/bin/bash
#
#$ -cwd
#$ -pe smp 6
#$ -l mem_free=10G
#$ -S /bin/bash
#$ -e /common/bermanblab/data/private_data/POC_ROC/POC_ROC_RNA-Seq/POC_ROC_phase1_bw/scratch
#$ -o /common/bermanblab/data/private_data/POC_ROC/POC_ROC_RNA-Seq/POC_ROC_phase1_bw/scratch

set -xe

INFILE="/common/bermanblab/data/private_data/POC_ROC/POC_ROC_RNA-Seq/POC_ROC_phase1_bw/$1.wig"
OUTFILE="$1.genome.sorted.bw"

[[ -e "${INFILE}" ]] && echo "infile ${INFILE} exists"

source /home/dabkek/miniconda3/etc/profile.d/conda.sh
conda activate /home/dabkek/miniconda3/envs/rnaseq

/common/coetzeesg/pocroc/scripts/wigToBigWig \
	${INFILE} \
	/common/coetzeesg/pocroc/reference/GRCh38.primary_assembly.RSEM/chrNameLength.txt \
	${OUTFILE} \
