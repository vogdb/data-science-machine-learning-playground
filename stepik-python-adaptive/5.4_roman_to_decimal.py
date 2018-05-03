import helper

lines = helper.read_file('5.4_test.txt')

number_rome = lines[0]

number_decimal_list = []
rome_to_decimal = {'M': 1000, 'D': 500, 'C': 100, 'L': 50, 'X': 10, 'V': 5, 'I': 1}
for char in number_rome:
    number_decimal_list.append(rome_to_decimal[char])
number_decimal = 0
for num1, num2 in zip(number_decimal_list, number_decimal_list[1:]):
    if num1 >= num2:
        number_decimal += num1
    else:
        number_decimal -= num1
number_decimal += num2

helper.write_stdout(str(number_decimal))
