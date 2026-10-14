#!/usr/bin/env Rscript
# Only adapts GEO's RDS storage to the existing project's scRNA reference builder.
# It performs no cell-type aggregation or reannotation.

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 5L) {
  stop("usage: Rscript scripts/convert_lung_gse131907_rds_to_10x.R <raw.rds> <annotation.txt.gz> <hd_features.tsv.gz> <output.h5> <output_metadata.tsv>")
}
if (!requireNamespace("Matrix", quietly=TRUE) || !requireNamespace("hdf5r", quietly=TRUE)) {
  stop("Matrix and hdf5r are required for the GEO RDS storage adapter")
}
annotation <- read.delim(gzfile(args[[2]]), check.names=FALSE, stringsAsFactors=FALSE)
required <- c("Index", "Sample_Origin", "Cell_type")
if (!all(required %in% names(annotation))) stop("GSE131907 author annotation missing required columns")
annotation <- annotation[
  annotation$Sample_Origin %in% c("tLung", "nLung", "tL/B") &
    !is.na(annotation$Cell_type) & annotation$Cell_type != "Undetermined",
  , drop=FALSE
]
if (anyDuplicated(annotation$Index)) stop("Duplicate author cell Index")
features <- read.delim(gzfile(args[[3]]), header=FALSE, stringsAsFactors=FALSE)
hd_genes <- unique(features[[2]])
cat("Reading published GSE131907 raw RDS; no annotation is inferred.\n", file=stderr())
raw <- readRDS(args[[1]])
if (!is.data.frame(raw) || is.null(rownames(raw)) || is.null(colnames(raw))) stop("Unexpected GEO RDS schema")
gene_idx <- match(hd_genes, rownames(raw))
gene_idx <- gene_idx[!is.na(gene_idx)]
genes <- rownames(raw)[gene_idx]
cell_idx <- match(annotation$Index, colnames(raw))
if (anyNA(cell_idx) || length(genes) < 1000L) stop("GEO cells or HD genes failed exact symbol matching")
cat(sprintf("Adapting %d genes x %d author-annotated cells.\n", length(genes), length(cell_idx)), file=stderr())
x <- Matrix::Matrix(as.matrix(raw[gene_idx, cell_idx, drop=FALSE]), sparse=TRUE)
if (!inherits(x, "dgCMatrix")) x <- methods::as(x, "dgCMatrix")
if (any(x@x < 0) || any(!is.finite(x@x))) stop("Invalid raw counts")
dir.create(dirname(args[[4]]), showWarnings=FALSE, recursive=TRUE)
f <- hdf5r::H5File$new(args[[4]], mode="w")
m <- f$create_group("matrix")
m[["data"]] <- x@x
m[["indices"]] <- x@i
m[["indptr"]] <- x@p
m[["shape"]] <- as.integer(dim(x))
m[["barcodes"]] <- as.character(annotation$Index)
g <- m$create_group("features")
g[["id"]] <- genes
g[["name"]] <- genes
g[["feature_type"]] <- rep("Gene Expression", length(genes))
f$close_all()
metadata <- data.frame(cell_id=annotation$Index, sct_cell_type=annotation$Cell_type)
write.table(metadata, args[[5]], sep="\t", quote=FALSE, row.names=FALSE)
cat(sprintf("Wrote 10x-compatible H5 with %d nonzero raw counts; aggregation belongs to the existing Python builder.\n", length(x@x)), file=stderr())
