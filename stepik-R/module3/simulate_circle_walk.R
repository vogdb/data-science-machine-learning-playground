simulate_walk <- function(radius = 6, n_max = 100, p = 1e-2) {
  current_position <- c(0, 0)
  for (i in 1:n_max) {
    is_absorbed <- rbinom(1, 1, p)
    if (is_absorbed) return(list(status = "Absorbed", 
                                 position = current_position, 
                                 steps = i))
    current_position <- c(current_position[1] + rnorm(1), current_position[2] + rnorm(1))
    if (sqrt(sum(current_position^2)) > radius) return(list(status = "breach", 
                                                      position = current_position, 
                                                      steps = i))
  }
  return(list(status = "Max steps reached", 
              position = current_position,
              steps = n_max))
}

result <- replicate(10000, simulate_walk(), simplify = FALSE)
result <- data.frame(
  status = sapply(result, function(x) x$status),
  position_x = sapply(result, function(x) x$position[1]),
  position_y = sapply(result, function(x) x$position[2]),
  steps = sapply(result, function(x) x$steps)
)

print(sum(result$status == "breach"))
