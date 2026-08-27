# Cross-check invariance/nrm.py against mirt's nominal response model on a fixture.
#
#   .venv/bin/python invariance/nrm.py --mirt-fixture invariance/out/mirt_fixture.csv
#   Rscript invariance/mirt_check.R invariance/out/mirt_fixture.csv
#
# mirt's DEFAULT itemtype="nominal" fixes ak for the first AND last category
# (0 and ncat-1), which is one constraint per item more than identification needs
# and makes it a restricted NRM. We free the last-category slope so the fitted model
# is Bock's general NRM, i.e. exactly what nrm.py fits.
suppressMessages(library(mirt))
args <- commandArgs(TRUE)
csv <- if (length(args)) args[1] else "invariance/out/mirt_fixture.csv"
d <- read.csv(csv)
K <- length(unique(unlist(d)))
p <- mirt(d, 1, itemtype = "nominal", pars = "values")
p$est[p$name == paste0("ak", K - 1)] <- TRUE          # free the top slope
m <- mirt(d, 1, itemtype = "nominal", pars = p, verbose = FALSE,
          technical = list(NCYCLES = 5000))
th <- matrix(seq(-4, 4, length.out = 61))
out <- do.call(cbind, lapply(seq_len(ncol(d)), function(i)
  probtrace(extract.item(m, i), th)))
write.csv(out, sub("\\.csv$", "_mirt_probs.csv", csv), row.names = FALSE)
cat("mirt logLik", extract.mirt(m, "logLik"),
    " npars", extract.mirt(m, "nest"), " items", ncol(d), " n", nrow(d), "\n")
