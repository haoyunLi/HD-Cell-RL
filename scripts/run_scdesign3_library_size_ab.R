#!/usr/bin/env Rscript
# Controlled generator-only experiment; existing source and fitted baseline are read-only.
suppressPackageStartupMessages({
  library(data.table)
  library(Matrix)
  library(rhdf5)
  library(SingleCellExperiment)
  library(scDesign3)
})

load_generator_helpers <- function(path, env = new.env(parent = globalenv())) {
  # Reuse helper implementations without executing the generator's main program.
  for (expr in parse(path)) {
    if (is.call(expr) && identical(expr[[1L]], as.name("<-")) &&
        length(expr) == 3L && is.call(expr[[3L]]) &&
        identical(expr[[3L]][[1L]], as.name("function"))) eval(expr, env)
  }
  env
}

draw_libraries <- function(labels, reference_labels, libraries, seed) {
  stopifnot(length(reference_labels) == length(libraries),
            all(is.finite(libraries)), all(libraries > 0),
            all(labels %in% reference_labels))
  set.seed(seed)
  source <- integer(length(labels))
  for (type in sort(unique(labels))) {
    candidates <- which(reference_labels == type)
    idx <- which(labels == type)
    source[idx] <- candidates[sample.int(length(candidates), length(idx), replace = TRUE)]
  }
  list(library = libraries[source], source_index = source)
}

expand_parameters <- function(parameters, types, labels, libraries = NULL, reference_library = NULL) {
  idx <- match(labels, types)
  stopifnot(!anyNA(idx))
  result <- lapply(parameters, function(x) if (is.null(x)) NULL else x[idx, , drop = FALSE])
  if (!is.null(libraries)) {
    stopifnot(length(libraries) == length(labels), all(libraries > 0),
              length(reference_library) == length(types), all(reference_library > 0))
    # Exact for the specified log-link offset; verified against extract_para below.
    result$mean_mat <- result$mean_mat * (libraries / reference_library[idx])
  }
  result
}

summarize_cells <- function(x, cell_ids, labels, method, seed, requested_library = NA_real_) {
  stopifnot(ncol(x) == length(cell_ids), length(labels) == length(cell_ids))
  data.table(cell_id = cell_ids, cell_type = labels, method = method, seed = seed,
             counts = as.numeric(Matrix::colSums(x)),
             detected_genes = as.numeric(Matrix::colSums(x > 0)),
             requested_library = requested_library)
}

gene_moments <- function(x, genes, labels, method, seed) {
  rbindlist(lapply(sort(unique(labels)), function(type) {
    z <- x[, labels == type, drop = FALSE]
    mu <- Matrix::rowMeans(z)
    variance <- pmax(0, Matrix::rowMeans(z ^ 2) - mu ^ 2) * ncol(z) / (ncol(z) - 1)
    data.table(gene = genes, cell_type = type, method = method, seed = seed,
               mean = mu, variance = variance, zero_fraction = Matrix::rowMeans(z == 0))
  }))
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 4L) stop("Usage: runner.R GENERATOR_SCRIPT BASELINE_DIR RAW_H5 OUTPUT_DIR [CORES=24] [SMOKE_GENES=0]")
  source_script <- normalizePath(args[1L], mustWork = TRUE)
  baseline_dir <- normalizePath(args[2L], mustWork = TRUE)
  h5 <- normalizePath(args[3L], mustWork = TRUE)
  out <- args[4L]
  cores <- if (length(args) >= 5L) as.integer(args[5L]) else 24L
  smoke_genes <- if (length(args) >= 6L) as.integer(args[6L]) else 0L
  stopifnot(cores > 0L, smoke_genes >= 0L)
  if (file.exists(out)) stop("Refusing to overwrite existing experiment: ", out)
  dir.create(out, recursive = TRUE)
  dir.create(file.path(out, "subdiagnostic"))
  helpers <- load_generator_helpers(source_script)
  log <- helpers$log_msg
  baseline_file <- file.path(baseline_dir, "simulation_model_state.rds")
  baseline <- readRDS(baseline_file)
  signature <- baseline$signature
  stopifnot(identical(signature$scdesign3_version, as.character(packageVersion("scDesign3"))),
            identical(signature$corr_formula, "1"), identical(signature$sigma_formula, "1"),
            identical(unname(tools::md5sum(h5)), signature$reference_h5_md5))
  RNGkind("L'Ecuyer-CMRG")
  set.seed(signature$seed)
  ids <- as.character(h5read(h5, "matrix/barcodes"))
  genes <- as.character(h5read(h5, "matrix/features/name"))
  features <- as.character(h5read(h5, "matrix/features/id"))
  shape <- as.integer(h5read(h5, "matrix/shape"))
  stopifnot(!anyDuplicated(ids), !anyDuplicated(genes))
  reference <- fread(file.path(baseline_dir, "model_reference_cell_metadata.tsv"))
  stopifnot(identical(reference$Cell, signature$sampled_cell_ids),
            identical(reference$cell_type, signature$sampled_cell_types))
  keep_cells <- match(reference$Cell, ids)
  stopifnot(!anyNA(keep_cells))
  selected_genes <- signature$model_gene_index
  if (smoke_genes > 0L) {
    audit <- fread(file.path(baseline_dir, "reference_gene_model_audit.csv"))
    selected_genes <- sort(head(selected_genes[order(audit$reference_mean[selected_genes], decreasing = TRUE)], smoke_genes))
  }
  full <- new("dgCMatrix", p = as.integer(h5read(h5, "matrix/indptr")),
              i = as.integer(h5read(h5, "matrix/indices")),
              x = as.numeric(h5read(h5, "matrix/data")), Dim = shape)
  x <- full[selected_genes, keep_cells, drop = FALSE]
  stopifnot(all(is.finite(x@x)), all(x@x >= 0), all(x@x == round(x@x)))
  libraries <- as.numeric(Matrix::colSums(x))
  rownames(x) <- genes[selected_genes]
  colnames(x) <- reference$Cell
  # Retain the raw full-panel reference statistics to expose filtering losses.
  fwrite(summarize_cells(full[, keep_cells], reference$Cell, reference$cell_type,
                         "reference_full_panel", 0L), file.path(out, "subdiagnostic/reference_full_panel.csv"))
  rm(full); gc()
  types <- signature$model_types
  target_labels <- reference$cell_type
  if (smoke_genes > 0L) target_labels <- rep(types, each = 3L)
  target_ids <- sprintf("synthetic_%07d", seq_along(target_labels))
  reference_library <- vapply(types, function(t) median(libraries[reference$cell_type == t]), numeric(1))
  stopifnot(all(libraries > 0), all(reference_library > 0))
  meta <- data.frame(cell_type = factor(reference$cell_type, levels = types), library = libraries,
                     row.names = reference$Cell)
  sce <- SingleCellExperiment(assays = list(counts = x), colData = meta)
  base_data <- construct_data(sce = sce, assay_use = "counts", celltype = "cell_type",
                              pseudotime = NULL, spatial = NULL, other_covariates = NULL,
                              ncell = ncol(x), corr_by = "1")
  offset_data <- construct_data(sce = sce, assay_use = "counts", celltype = "cell_type",
                                pseudotime = NULL, spatial = NULL, other_covariates = "library",
                                ncell = ncol(x), corr_by = "1")
  gene_positions <- match(selected_genes, signature$model_gene_index)
  a_parameters <- lapply(baseline$type_parameters, function(z) if (is.null(z)) NULL else z[, gene_positions, drop = FALSE])
  a_copula <- lapply(baseline$copula_list, function(z) z[gene_positions, gene_positions, drop = FALSE])
  important <- baseline$important_feature[gene_positions]
  protocol <- list(experiment = "scDesign3_library_size_AB", phase = if (smoke_genes) "smoke" else "development",
                   reference_model_dir = baseline_dir, source_generator = source_script,
                   source_generator_md5 = unname(tools::md5sum(source_script)),
                   baseline_model_md5 = unname(tools::md5sum(baseline_file)),
                   raw_h5 = h5, raw_h5_md5 = signature$reference_h5_md5,
                   scdesign3_version = as.character(packageVersion("scDesign3")),
                   n_reference_cells = ncol(x), n_modeled_genes = nrow(x), n_types = length(types),
                   n_synthetic_cells_per_method_seed = length(target_labels), n_cores = cores,
                   mu_A = "cell_type", mu_B = "cell_type + offset(log(library))",
                   sigma_formula = "1", corr_formula = "1", family = "nb",
                   library_definition = "sum of modeled-gene raw counts, drawn within training cell type with replacement",
                   fit_seed = signature$seed, simulation_seeds = if (smoke_genes) 20260916L else 20260916:20260918,
                   important_feature_threshold = signature$important_feature,
                   same_reference_cells = TRUE, same_gene_axis = TRUE, same_target_type_composition = TRUE,
                   target_depth_is_not_exact_count_constraint = TRUE,
                   no_EM_RL_or_capture_changes = TRUE, heldout_donor_used_for_fitting = FALSE,
                   heldout_donor_evaluated = FALSE)
  jsonlite::write_json(protocol, file.path(out, "protocol.json"), pretty = TRUE, auto_unbox = TRUE)
  file.copy(source_script, file.path(out, "subdiagnostic/generator_source.R"), overwrite = FALSE)
  fwrite(reference, file.path(out, "subdiagnostic/reference_cells.tsv"), sep = "\t")
  fwrite(data.table(gene = genes[selected_genes], reference_index = selected_genes),
         file.path(out, "subdiagnostic/modeled_genes.csv"))
  fwrite(summarize_cells(x, reference$Cell, reference$cell_type, "reference", 0L),
         file.path(out, "reference_per_cell.csv"))
  fwrite(gene_moments(x, genes[selected_genes], reference$cell_type, "reference", 0L),
         file.path(out, "subdiagnostic/reference_gene_moments.csv.gz"))
  log("Fit library-offset marginals:", nrow(x), "genes,", ncol(x), "reference cells")
  set.seed(signature$seed)
  fit <- fit_marginal(data = offset_data, predictor = "gene",
                      mu_formula = protocol$mu_B, sigma_formula = "1", family_use = "nb",
                      n_cores = cores, usebam = FALSE, edf_flexible = FALSE,
                      parallelization = "mcmapply", trace = FALSE, simplify = TRUE)
  log("Fit offset residual copula with unchanged feature selection / sparsification")
  b_copula <- helpers$fit_gaussian_copula_for_simulation(
    sce = sce, assay_use = "counts", input_data = offset_data$dat, marginal_list = fit,
    family_use = "nb", important_feature = important, n_cores = cores)
  cov <- data.frame(cell_type = factor(types, levels = types), library = reference_library,
                    corr_group = 1, row.names = paste0("type_", seq_along(types)))
  parameters <- extract_para(sce = sce, assay_use = "counts", marginal_list = fit,
                             n_cores = cores, family_use = "nb", new_covariate = cov,
                             parallelization = "mcmapply", data = offset_data$dat)
  # Verify cached offset expansion against direct prediction at two depths per type.
  check_labels <- rep(types, each = 2L)
  check_libraries <- rep(reference_library, each = 2L) * rep(c(0.5, 2), length(types))
  check_cov <- data.frame(cell_type = factor(check_labels, levels = types), library = check_libraries,
                          corr_group = 1, row.names = paste0("check_", seq_along(check_labels)))
  direct <- extract_para(sce = sce, assay_use = "counts", marginal_list = fit,
                         n_cores = cores, family_use = "nb", new_covariate = check_cov,
                         parallelization = "mcmapply", data = offset_data$dat)
  expanded <- expand_parameters(parameters, types, check_labels, check_libraries, reference_library)
  stopifnot(isTRUE(all.equal(unname(direct$mean_mat), unname(expanded$mean_mat), tolerance = 1e-6)),
            isTRUE(all.equal(unname(direct$sigma_mat), unname(expanded$sigma_mat), tolerance = 1e-6)))
  saveRDS(list(parameters = parameters, copula_list = b_copula$copula_list,
               reference_library = reference_library, important_feature = important,
               selected_genes = selected_genes, protocol = protocol),
          file.path(out, "offset_model.rds"), compress = FALSE)
  rm(fit, direct, expanded, b_copula, baseline); gc()
  offset_model <- readRDS(file.path(out, "offset_model.rds"))
  for (seed in protocol$simulation_seeds) {
    draws <- draw_libraries(target_labels, reference$cell_type, libraries, seed + 100000L)
    fwrite(data.table(cell_id = target_ids, cell_type = target_labels, library = draws$library,
                       source_cell_id = reference$Cell[draws$source_index]),
           file.path(out, "subdiagnostic", paste0("library_draws_", seed, ".csv")))
    for (method in c("A_current", "B_library_offset")) {
      is_b <- identical(method, "B_library_offset")
      p <- expand_parameters(if (is_b) parameters else a_parameters, types, target_labels,
                             if (is_b) draws$library else NULL, if (is_b) reference_library else NULL)
      rownames(p$mean_mat) <- target_ids
      rownames(p$sigma_mat) <- target_ids
      if (!is.null(p$zero_mat)) rownames(p$zero_mat) <- target_ids
      new_cov <- data.frame(cell_type = factor(target_labels, levels = types),
                            library = draws$library, corr_group = 1, row.names = target_ids)
      if (!is_b) new_cov$library <- NULL
      log("Simulate", method, "seed", seed, "cells", length(target_ids))
      set.seed(seed)
      simulated <- simu_new(sce = sce, assay_use = "counts", mean_mat = p$mean_mat,
                             sigma_mat = p$sigma_mat, zero_mat = p$zero_mat, quantile_mat = NULL,
                             copula_list = if (is_b) offset_model$copula_list else a_copula,
                             n_cores = cores, fastmvn = FALSE, family_use = "nb",
                             nonnegative = TRUE, nonzerovar = FALSE,
                             input_data = if (is_b) offset_data$dat else base_data$dat,
                             new_covariate = new_cov, important_feature = important,
                             parallelization = "mcmapply", filtered_gene = base_data$filtered_gene)
      stopifnot(all(is.finite(simulated)), all(simulated >= 0), all(simulated == round(simulated)),
                nrow(simulated) == nrow(x), ncol(simulated) == length(target_ids))
      prefix <- paste0(method, "_", seed)
      fwrite(summarize_cells(simulated, target_ids, target_labels, method, seed,
                             if (is_b) draws$library else NA_real_), file.path(out, paste0(prefix, "_per_cell.csv")))
      fwrite(gene_moments(simulated, genes[selected_genes], target_labels, method, seed),
             file.path(out, "subdiagnostic", paste0(prefix, "_gene_moments.csv.gz")))
      if (seed == protocol$simulation_seeds[1L]) {
        helpers$write_chunk_h5(simulated, selected_genes, features, genes, target_ids,
                               target_labels, digest::digest(list(protocol, method, seed)),
                               file.path(out, paste0(prefix, ".h5")))
      }
      rm(simulated, p); gc()
    }
  }
  writeLines(capture.output(sessionInfo()), file.path(out, "subdiagnostic/sessionInfo.txt"))
  jsonlite::write_json(list(completed = TRUE, offset_prediction_check = TRUE,
                            methods = c("A_current", "B_library_offset"), seeds = protocol$simulation_seeds),
                       file.path(out, "completion.json"), pretty = TRUE, auto_unbox = TRUE)
  log("COMPLETE", out)
}

if (sys.nframe() == 0L) main()
