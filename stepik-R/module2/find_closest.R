find_closest <- function(v, n) {
  test <- abs(v - n)
  which(test == min(test))
}


v <- c(5, 2, 7, 7, 7, 2, 0, 0)
n = 1

print(find_closest(v, n))