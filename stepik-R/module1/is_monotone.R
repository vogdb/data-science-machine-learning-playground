is_monotone <- function(x) {
  next_item = x[-1]
  prev_item = x[-length(x)]
  diff = next_item - prev_item
  result = (sum(diff >= 0) == length(diff)) || (sum(diff <= 0) == length(diff))
  return(result)
}


x=c(0, 0, 3, 4, 4, 8)
y=c(3:0, 1)

print(is_monotone(x))
print(is_monotone(y))