library(stringi)

decorate_string <- function(pattern, ...) { 
  pattern_r = stri_reverse(pattern)
  tmp = paste(...)
  paste0(pattern, tmp, pattern_r)
}



# "123abc321"
decorate_string(pattern = "123", "abc")
# "123abc def321"
decorate_string(pattern = "123", "abc", "def")    
# "123abc321" "123def321"
decorate_string(pattern = "123", c("abc", "def")) 


# "123abc+def321"
decorate_string(pattern = "123", "abc", "def", sep = "+") 
# "!x_x!"
decorate_string(pattern = "!", c("x", "x"), collapse = "_") 
# ".:1&3&5:." ".:2&4&6:."
decorate_string(pattern = ".:", 1:2, 3:4, 5:6, sep = "&")    