

######################################################
##    DATA MATRIX FOR HRDETECT FXN                  ##
######################################################

sample_names <- c("22421-1671-Omm",
               "22421-1671-RtOv",
               "27481-2604-LtOv",
               "27481-2604-RtOv",
               "41245-3489",
               "22421-2105-C",
               "22421-3793",
               "27481-3293",
               "27481-4016",
               "41245-4416-LtLN",
               "41245-4416-RtLN"
               )


data_matrix_hrdetect1 <- read.delim("~/data_matrix_hrdetect1.txt", row.names=1)



######################################################
##   SVs - POINT TO FILES                           ##
######################################################

## point HRDetect to Correct Saved File Location
SV_bedpe_files <- c("~/POCROC_bedpes/22421-1671-Omm_gridss.somatic.bedpe",
               "~/POCROC_bedpes/22421-1671-RtOv_gridss.somatic.bedpe",
               "~/POCROC_bedpes/27481-2604-LtOv_gridss.somatic.bedpe",
               "~/POCROC_bedpes/27481-2604-RtOv_gridss.somatic.bedpe",
               "~/POCROC_bedpes/41245-3489_gridss.somatic.bedpe",
               "~/POCROC_bedpes/22421-2105-C_gridss.somatic.bedpe",
               "~/POCROC_bedpes/22421-3793_gridss.somatic.bedpe",
               "~/POCROC_bedpes/27481-3293_gridss.somatic.bedpe",
               "~/POCROC_bedpes/27481-4016_gridss.somatic.bedpe",
               "~/POCROC_bedpes/41245-4416-LtLN_gridss.somatic.bedpe",
               "~/POCROC_bedpes/41245-4416-RtLN_gridss.somatic.bedpe"
               )

names(SV_bedpe_files) <- sample_names


######################################################
##   INDELS - POINT TO FILES                        ##
######################################################


Indels_vcf_files <- c("~/POCROC_Mutect2_Indels/22421-1671-Omm_Mutect2-filt_INDELs.vcf.gz",
                        "~/POCROC_Mutect2_Indels/22421-1671-RtOv_Mutect2-filt_INDELs.vcf.gz",
                        "~/POCROC_Mutect2_Indels/27481-2604-LtOv_Mutect2-filt_INDELs.vcf.gz",
                        "~/POCROC_Mutect2_Indels/27481-2604-RtOv_Mutect2-filt_INDELs.vcf.gz",
                        "~/POCROC_Mutect2_Indels/41245-3489_Mutect2-filt_INDELs.vcf.gz",
                        "~/POCROC_Mutect2_Indels/22421-2105-C_Mutect2-filt_INDELs.vcf.gz",
                        "~/POCROC_Mutect2_Indels/22421-3793_Mutect2-filt_INDELs.vcf.gz",
                        "~/POCROC_Mutect2_Indels/27481-3293_Mutect2-filt_INDELs.vcf.gz",
                        "~/POCROC_Mutect2_Indels/27481-4016_Mutect2-filt_INDELs.vcf.gz",
                        "~/POCROC_Mutect2_Indels/41245-4416-LtLN_Mutect2-filt_INDELs.vcf.gz",
                        "~/POCROC_Mutect2_Indels/41245-4416-RtLN_Mutect2-filt_INDELs.vcf.gz"
                      )
names(Indels_vcf_files) <- sample_names





######################################################
##   COPY NUMBER - POINT TO FILES                   ##
######################################################

## point HRDetect to Correct Saved File Location
CNV_tab_files <- c("~/POCROC_CNV/copy_number_22421-1671-Omm.txt",
                   "~/POCROC_CNV/copy_number_22421-1671-RtOv.txt",
                   "~/POCROC_CNV/copy_number_27481-2604-LtOv.txt",
                   "~/POCROC_CNV/copy_number_27481-2604-RtOv.txt",
                   "~/POCROC_CNV/copy_number_41245-3489.txt",
                   "~/POCROC_CNV/copy_number_22421-2105-C.txt",
                   "~/POCROC_CNV/copy_number_22421-3793.txt",
                   "~/POCROC_CNV/copy_number_27481-3293.txt",
                   "~/POCROC_CNV/copy_number_27481-4016.txt",
                   "~/POCROC_CNV/copy_number_41245-4416-LtLN.txt",
                   "~/POCROC_CNV/copy_number_41245-4416-RtLN.txt"
                    )
names(CNV_tab_files) <- sample_names


######################################################
##   RUN HR DETECT FUNCTION                         ##
######################################################

library(signature.tools.lib)
res <- HRDetect_pipeline(data_matrix_hrdetect1,   #or data_matrix_hrdetect1 for multiple samples as noted above
                         genome.v = "hg38",
                         SNV_tab_files = NULL,
                         SV_bedpe_files = SV_bedpe_files,
                         Indels_vcf_files = Indels_vcf_files,
                         CNV_tab_files = CNV_tab_files,
                         methodFit = "NNLS",
                         bootstrapSignatureFit = FALSE,
                         threshold_percentFit = 10,
                         nparallel = 1)


