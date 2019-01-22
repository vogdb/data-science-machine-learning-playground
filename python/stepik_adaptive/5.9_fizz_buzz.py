import helper
import numpy as np

lines = helper.read_file('5.9_test.txt')
left, right = [int(num) for num in lines[0].split(' ')]
sequence = np.array(range(left, right + 1))

test_sequence_3 = sequence % 3
test_sequence_5 = sequence % 5
test_sequence_15 = sequence % 15
sequence = sequence.astype(str)
sequence[test_sequence_3 == 0] = 'Fizz'
sequence[test_sequence_5 == 0] = 'Buzz'
sequence[test_sequence_15 == 0] = 'FizzBuzz'

helper.write_stdout('\n'.join(sequence))


# Sample Input:
#
# 8 16
# Sample Output:
#
# 8
# Fizz
# Buzz
# 11
# Fizz
# 13
# 14
# FizzBuzz
# 16
