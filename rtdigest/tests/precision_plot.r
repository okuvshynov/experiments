require(ggplot2)

args = commandArgs(trailingOnly=TRUE)

df <- read.table(args[1], sep = " ", header = T)

ds <- split(df, df$dist)

for (d in ds) {
  g <- ggplot(d, aes(x=factor(scale_power), y=width, fill=factor(clusters))) +
    geom_boxplot(outlier.shape=NA) +
    facet_grid(factor(quantile) ~ factor(scale_quantile), scales="fixed")
  png(file = sprintf("tests/data/precision_%s.png", d$dist), width=2048, height=1024)
  print(g)
}
