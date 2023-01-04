#!/bin/bash
#
#$ -cwd
#$ -pe smp 6
#$ -l mem_free=10G
#
#$ -S /bin/bash
#$ -e path_to_working_dir
#$ -o path_to_working_dir

grep -w "chr2" $1.bedgraph > $1.chr2.bedgraph
