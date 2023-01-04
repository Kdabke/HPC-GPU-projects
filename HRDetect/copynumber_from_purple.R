######################################################
##   COPY NUMBER - GENERATE FROM PURPLE OUTPUT      ##
######################################################
args <- commandArgs(trailingOnly = TRUE)
fn <- args[1]
fn2 <- args[2]


df <- read.delim(paste0(fn))


# first column needs to be unique ID
copy_number <- as.data.frame(seq(1:nrow(df)))

# need to drop chr text from file
x <- stringr::str_split(df[,1],
                        pattern = "chr")
x <- sapply(x, function(X) {
  X <- X[2]
  X <- paste(X, collapse = "")
})

copy_number$Chromosome <- x

# bind chromosome positions
copy_number <- cbind(copy_number,df[,2:3])

# manually add normal copy number columns --> 2 total copies for all samples, 1 for minor copy number
copy_number$total.copy.number.inNormal <- 2
copy_number$minor.copy.number.inNormal <- 1

# bind values from purple
copy_number$total.copy.number.inTumour <- df[,4]
copy_number$minor.copy.number.inTumour <- df[,15]

# hrdetect needs these columnn names exactly
colnames(copy_number)[1:4] <- c("seg_no","Chromosome","chromStart","chromEnd")

# will only work with rounded numbers for copy number info
copy_number$total.copy.number.inTumour <- round(copy_number$total.copy.number.inTumour)
copy_number$minor.copy.number.inTumour <- round(copy_number$minor.copy.number.inTumour)

write.table(copy_number, paste0(fn2),
            quote=FALSE,
            sep='\t',
            row.names=FALSE,
            col.names=T)
