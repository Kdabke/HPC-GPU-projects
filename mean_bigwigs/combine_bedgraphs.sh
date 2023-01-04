#!/bin/bash
#
#$ -cwd
#$ -pe smp 6
#$ -l mem_free=10G
#
#$ -S /bin/bash
#$ -e /common/bermanblab/data/private_data/POC_ROC/POC_ROC_RNA-Seq/POC_ROC_phase1_bw/bigwigs/bedgraph_HRD
#$ -o /common/bermanblab/data/private_data/POC_ROC/POC_ROC_RNA-Seq/POC_ROC_phase1_bw/bigwigs/bedgraph_HRD


conda activate mean_bigwigs

bedtools unionbedg -i *.chr2.bedgraph | awk 'OFS="\t" {sum=0; for (col=4; col<=NF; col++) sum += $col; print $1, $2, $3, sum/(NF-4+1); }' > temp_mean_chr2.bedgraph
