source("scripts/run_scdesign3_library_size_ab.R")
RNGkind("L'Ecuyer-CMRG")
labels <- c("A", "B", "A", "B")
draw <- draw_libraries(labels, c("A", "A", "B"), c(100, 300, 900), 7L)
stopifnot(all(draw$library[labels == "A"] %in% c(100, 300)),
          all(draw$library[labels == "B"] == 900),
          identical(draw, draw_libraries(labels, c("A", "A", "B"), c(100, 300, 900), 7L)))
expect_error <- function(expr) stopifnot(inherits(tryCatch(force(expr), error = identity), "error"))
expect_error(draw_libraries("C", "A", 100, 7L))
expect_error(draw_libraries("A", "A", 0, 7L))
p <- list(mean_mat = matrix(c(1, 2, 3, 4), nrow = 2), sigma_mat = matrix(5, 2, 2), zero_mat = NULL)
unchanged <- expand_parameters(p, c("A", "B"), labels)
stopifnot(identical(unchanged$mean_mat, p$mean_mat[c(1,2,1,2), ]))
changed <- expand_parameters(p, c("A", "B"), labels, c(50, 400, 200, 200), c(100, 200))
stopifnot(isTRUE(all.equal(changed$mean_mat, unchanged$mean_mat * c(.5, 2, 2, 1))),
          identical(changed$sigma_mat, unchanged$sigma_mat), is.null(changed$zero_mat))
expect_error(expand_parameters(p, c("A", "B"), "C"))
x <- Matrix(matrix(c(0, 1, 2, 3, 0, 5), nrow = 3), sparse = TRUE)
stats <- summarize_cells(x, c("a", "b"), c("A", "A"), "fixture", 1L)
stopifnot(identical(stats$counts, c(3, 8)), identical(stats$detected_genes, c(2, 2)))
moments <- gene_moments(x, c("g1", "g2", "g3"), c("A", "A"), "fixture", 1L)
stopifnot(isTRUE(all.equal(moments$variance, apply(as.matrix(x), 1, var))),
          isTRUE(all.equal(moments$zero_fraction, rowMeans(as.matrix(x) == 0))))
cat("PASS: library draws, invalid inputs, unchanged A, offset scaling, cell and gene statistics\n")
