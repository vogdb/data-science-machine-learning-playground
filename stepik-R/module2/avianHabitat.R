# also see https://github.com/tonytonov/Rcourse/blob/master/R%20programming/avianHabitat.R
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

avian <- read.csv("avianHabitat.csv")

#avian2 <- read.csv("avianHabitat2.csv", skip = 5, sep = ';', comment.char = '%', na.strings = c("Don't remember"))
#avian2$Observer <- 'KL'
# reorder as avian
#avian2 <- avian2[, colnames(avian)]
# merge together
#avian <- rbind(avian, avian2)


print(head(avian))

# Transforming variables

height_variables <- names(avian)[-(1:5)][c(T, F)]
print(height_variables)
# max values of Height columns
ff = apply(avian[,height_variables], 2, function(x) max(x, na.rm = TRUE))
print(sort(ff))