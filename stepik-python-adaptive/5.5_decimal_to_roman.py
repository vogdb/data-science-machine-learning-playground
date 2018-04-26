import helper

lines = helper.read_file('5.5_test.txt')

number = int(lines[0])
assert 0 < number < 4000, 'invalid range'

hundreds = ["", 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM']
tens = ["", 'X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC']
ones = ["", 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX']


number_rome = ''
if number >= 1000:
    thousands = number // 1000
    number = number % 1000
    number_rome = 'M' * thousands

number_rome += hundreds[number // 100]
number = number % 100
number_rome += tens[number // 10]
number = number % 10
number_rome += ones[number]

print(number_rome)

# 1984
# MCMLXXXIV
