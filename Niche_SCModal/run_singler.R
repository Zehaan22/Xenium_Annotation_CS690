#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(SingleR)
    library(SummarizedExperiment)
})

args <- commandArgs(trailingOnly=TRUE)
tmpdir <- args[1]

query_file <- file.path(tmpdir, "query.csv")
ref_file   <- file.path(tmpdir, "ref.csv")
labels_file <- file.path(tmpdir, "ref_labels.csv")

# READ
sc_data  <- as.matrix(read.csv(query_file, row.names=1, check.names=FALSE))
ref_data <- as.matrix(read.csv(ref_file,   row.names=1, check.names=FALSE))
types    <- read.csv(labels_file, header=FALSE, stringsAsFactors=FALSE)[[1]]

# Build Reference SE object
ref_se <- SummarizedExperiment(
    assays=list(counts=ref_data),
    colData=data.frame(label=types)
)

# Run full SingleR (fine tuning included)
pred <- SingleR(
    sc_data=sc_data,
    ref_data=ref_data,
    types=types,
    fine.tune=TRUE
)

write.csv(pred$labels, file.path(tmpdir, "predictions.csv"), row.names=FALSE)
