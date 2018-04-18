get_fractions <- function(n, m) {
  u <- seq(0, 1, 1/n)
  v <- seq(0, 1, 1/m)
  r <- c(u, v)
  r <- sort(r, decreasing = TRUE)
  r <- unique(r)
  return(r)
}

print(get_fractions(3, 7))
