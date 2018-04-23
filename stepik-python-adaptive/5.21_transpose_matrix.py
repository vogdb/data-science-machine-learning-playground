import helper
import numpy as np

# lines = helper.read_stdin()
lines = helper.read_file('5.21_test.txt')

shape = lines[0]
rows = [list(line.split(' ')) for line in lines[1:]]
matrix = np.array(rows)

shape = shape.split(' ')
assert len(shape) == 2, 'the argument shape must be equal to 2'
shape = list(map(int, shape))
assert list(matrix.shape) == shape, 'the argument shape is not the same as the actual shape'

join_cols = map(lambda row: ' '.join(row) + '\n', matrix.T)
join_rows = ''.join(list(join_cols))
helper.write_stdout(join_rows)


