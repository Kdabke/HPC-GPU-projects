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

INFILE="/common/bermanblab/data/private_data/POC_ROC/coetzeesg_pocroc/phase2/rsem_trimmed/$1.bw"
OUTFILE1="$1.bedgraph"
OUTFILE2="$1.chr15.bedgraph"

[[ -e "${INFILE}" ]] && echo "infile ${INFILE} exists"


conda activate mean_bigwigs

bigWigToBedGraph ${INFILE} ${OUTFILE1}

grep "chr15" ${OUTFILE1} > ${OUTFILE2}
