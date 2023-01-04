######################################################
##   SV DATA -- GENERATE BEDPE FROM SV VCF          ##
######################################################
args <- commandArgs(trailingOnly = TRUE)
fn <- args[1]
fn2 <- args[2]
fn3 <- args[3]



library(StructuralVariantAnnotation)
library(stringr)
library(tidyr)

vcf <- readVcf(paste0(fn))
gr <- breakpointRanges(vcf)

bedpe <- data.frame(
  chrom1=seqnames(gr),
  start1=start(gr) - 1,
  end1=end(gr),
  chrom2=seqnames(partner(gr)),
  start2=start(partner(gr)) - 1,
  end2=end(partner(gr)),
  sv_id = gr$sourceId,
  strand1=strand(gr),
  strand2=strand(partner(gr)),
  #svclass=names(gr),
  sample = paste0(fn2)
  #score=gr$QUAL,)
)

write.table(bedpe, paste0(fn3),
            quote=FALSE,
            sep='\t',
            row.names=FALSE,
            col.names=T)
