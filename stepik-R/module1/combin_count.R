my_factorial <- function(n) {
  if(n == 0){
    return(1)
  } else {
    return(n*my_factorial(n-1))
  }
}

combin_count <- function(n, k, with_repretitions = FALSE) {
  if(with_repretitions){
    result = my_factorial(n + k - 1)/(my_factorial(k)*my_factorial(n-1))
  } else {
    result = my_factorial(n)/(my_factorial(k)*my_factorial(n-k))
  }
  return(floor(result))
}


print(combin_count(5, 3))
