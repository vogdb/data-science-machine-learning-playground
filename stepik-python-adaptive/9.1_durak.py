import helper

lines = helper.read_file('9.1_test.txt')
cards = lines[0].split(' ')
trump_suit = lines[1]

card1, card2 = [[card[:-1], card[-1]] for card in cards]

suits = ['C', 'D', 'H', 'S']
values = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

result = 'Error'
value1 = values.index(card1[0])
value2 = values.index(card2[0])
suit1 = card1[1]
suit2 = card2[1]

if suit1 == suit2:
    if value1 > value2:
        result = 'First'
    elif value2 > value1:
        result = 'Second'
elif suit1 == trump_suit:
    result = 'First'
elif suit2 == trump_suit:
    result = 'Second'

helper.write_stdout(result)

# Sample Input 1:
# AH JH
# D

# Sample Output 1:
# First
