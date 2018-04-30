values <- c("Ace", 2:10, "Jack", "Queen", "King")
suits <- c("Clubs", "Diamonds", "Hearts", "Spades")
card_deck <- outer(values, suits, paste, sep = " of ")
roulette_values <- c("Zero!", 1:36)

generator <- function(set, prob = rep(1/length(set), length(set))) { 
  function(n) sample(set, n, replace = T, prob = prob)
}


card_generator <- generator(card_deck)
coin_generator <- generator(c("Heads", "Tails"))
fair_roulette <- generator(roulette_values)

n = length(roulette_values)
rigged_prob <- c(2/(n+2), rep(1/(n+2), n - 1))
rigged_roulette <- generator(roulette_values, prob = rigged_prob)

# card_generator(10)
# coin_generator(5)

print(fair_roulette(50))
print(rigged_roulette(50))