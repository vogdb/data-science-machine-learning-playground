import helper

lines = helper.read_file('3.6_test.txt')
n = int(lines[0])

result = []
for i in range(1, n + 1):
    result = result + ([str(i)] * i)
    if len(result) > n:
        break

# n = 7
# 1 2 2 3 3 3 4
helper.write_stdout(' '.join(result[:n]))
