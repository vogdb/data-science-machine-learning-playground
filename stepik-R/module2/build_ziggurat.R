build_ziggurat <- function(n) {
  s = 2*n-1
  m = matrix(0, nrow = s, ncol = s)
  for(i in 0:(n-1)) {
    m[(1 + i):(s - i), (1 + i):(s - i)] <- i+1
  }
  m
}


# n = 1
# 1
# print(build_ziggurat(1))
print(build_ziggurat(1))

# n = 2
# 1   1   1
# 1   2   1
# 1   1   1


# n = 3
# 1   1   1   1   1
# 1   2   2   2   1
# 1   2   3   2   1
# 1   2   2   2   1
# 1   1   1   1   1

# n = 4
# 1   1   1   1   1   1   1
# 1   2   2   2   2   2   1
# 1   2   3   3   3   2   1
# 1   2   3   4   3   2   1
# 1   2   3   3   3   2   1
# 1   2   2   2   2   2   1
# 1   1   1   1   1   1   1
print(build_ziggurat(4))