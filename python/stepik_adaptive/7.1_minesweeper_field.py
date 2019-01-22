import helper
import numpy as np
# from scipy import signal


lines = helper.read_file('7.1_test.txt')

shape = lines[0]
rows = [list(line) for line in lines[1:]]

# prepare input
mine_field = np.array(rows)
mine_field[mine_field == '.'] = 0
mine_field[mine_field == '*'] = 1
mine_field = mine_field.astype(int)


mine_field_pad = np.pad(mine_field, pad_width=1, mode='constant')
mine_field_counted = np.zeros(mine_field.shape)
m, n = mine_field.shape
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if mine_field_pad[i][j] != 1:
            tmp = np.sum(mine_field_pad[(i - 1):(i + 2), (j - 1):(j + 2)])
            mine_field_counted[i - 1][j - 1] = tmp


# count mines by convolution
# conv_filter = np.array([
#     [1, 1, 1],
#     [1, 0, 1],
#     [1, 1, 1]
# ])
# mine_field_counted = signal.convolve2d(mine_field, conv_filter, mode='same')

mine_field_counted = mine_field_counted.astype(int).astype(str)
mine_field_counted[mine_field == 1] = '*'

join_cols = map(lambda row: ''.join(row) + '\n', mine_field_counted)
join_rows = ''.join(list(join_cols))
helper.write_stdout(join_rows)


# input:
#
# 4 4
# ..*.
# **..
# ..*.
# ....
#
# output:
#
# 23*1
# **32
# 23*1
# 0111
