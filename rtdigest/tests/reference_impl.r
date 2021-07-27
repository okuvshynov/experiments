args = commandArgs(trailingOnly=TRUE)

lb <- strtoi(args[1])
m <- strtoi(args[2])

qq = c(500000, 950000, 990000, 999000, 999900)

for (i in 1:m) {
  n <- sample(lb:(lb*2), 1)
  y <- rnorm(n)
  q <- quantile(y, qq / 1000000.0, type=1)

  for (j in 1:length(qq)) {
    cat(n)
    cat("\n")

    cat(qq[j])
    cat("\n")

    cat(q[j])
    cat("\n")

    cat(y)
    cat("\n")
  }
}
