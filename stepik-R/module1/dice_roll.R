dice_roll <- function(n) {
  x <- runif(n, min = 1, max = 6 + 1)
  x <- floor(x)
  return(x)
}

print(dice_roll(10))

