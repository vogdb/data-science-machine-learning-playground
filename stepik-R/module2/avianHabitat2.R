this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)
library(stringr)
options(stringsAsFactors = F)


avian <- read.csv("avianHabitat.csv")
avian$Observer <- factor(avian$Observer)

coverage_variables <- names(avian)[str_detect(names(avian), "^P")]
avian$total_coverage <- rowSums(avian[, coverage_variables])
avian$site_name <- factor(str_replace(avian$Site, "[:digit:]+", ""))

sites_coverage = tapply(avian$total_coverage, avian$site_name, mean)
print(sites_coverage)

height_variables <- names(avian)[str_detect(names(avian), "Ht$")]
max_height_by_observer = aggregate(avian[,height_variables], by=list(avian$Observer), max)
print(max_height_by_observer)
