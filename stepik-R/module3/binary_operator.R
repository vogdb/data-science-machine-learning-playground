"%+%" <- function(x, y) {
  l <- max(length(x), length(y))
  length(x) <- l
  length(y) <- l
  x + y
}


print(1:5 %+% 1:2)
# c(2, 4, NA, NA, NA)

print(5 %+% c(2, 6))
# c(7, NA) 