#!/bin/bash
#
#$ -cwd
#$ -pe smp 6
#$ -l mem_free=10G
#
#$ -S /bin/bash
#$ -e /common/bermanblab/data/private_data/POC_ROC/POC_ROC_RNA-Seq/POC_ROC_phase1_bw/bigwigs/bedgraph_HRD
#$ -o /common/bermanblab/data/private_data/POC_ROC/POC_ROC_RNA-Seq/POC_ROC_phase1_bw/bigwigs/bedgraph_HRD

grep -w "chr2" $1.bedgraph > $1.chr2.bedgraph
