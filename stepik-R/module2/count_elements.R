count_elements <- function(x){
  x_uniq = sort(unique(x))
  result <- sapply(x_uniq, function(x_i){
    print(x_i)
    c(x_i, length(which(x == x_i)))
  })
  return(result)
}


x <- c(5, 2, 7, 7, 7, 2, 0, 0)
count_elements(x)
# 0 2 5 7
# 2 2 1 3