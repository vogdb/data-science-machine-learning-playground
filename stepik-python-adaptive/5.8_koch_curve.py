import helper
import numpy as np
import turtle

lines = helper.read_file('5.8_test.txt')
n = int(lines[0])

assert 0 < n <= 10, 'invalid range of n'


def koch(base_angle, n):
    curve = np.array([0 + base_angle, 60, -120, 60])
    n -= 1
    if n > 0:
        curve = np.array([koch(turn, n) for turn in curve]).ravel()
    return curve


result = koch(0, n)[1:]
result = result.astype(str)
result = ['turn ' + item for item in result]
print('\n'.join(result))


######## UI test #########
def koch_turns(n):
    return koch(0, n)[1:]


def turtle_koch_curve(n, line_length=20):
    for move in koch_turns(n):
        turtle.forward(line_length)
        turtle.left(move)
    turtle.forward(line_length)


turtle_koch_curve(n)
turtle.done()
