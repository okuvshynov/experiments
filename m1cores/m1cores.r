require(ggplot2)

args = commandArgs(trailingOnly=TRUE)

d <- read.table(args[1], sep = ",", header = F, col.names=c("core", "unroll", "counter", "iterations", "total_events", "events"))
g <- ggplot(d, aes(x=factor(core), y=events, fill=factor(unroll))) +
  geom_boxplot(outlier.shape=NA) +
  facet_grid(factor(d$counter), scales="free")
png(file = "_out/m1cores.png", width=1024, height=1024)
print(g)
