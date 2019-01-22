import helper

lines = helper.read_file('5.15_test.txt')

fraction = lines[0].split('/')
numerator = int(fraction[0])
denominator = int(fraction[1])


def simplified_form(numerator, denominator):
    floor_part = numerator // denominator
    coef_list = [floor_part]
    fraction_part = numerator % denominator
    if fraction_part > 0:
        coef_list = coef_list + simplified_form(denominator, fraction_part)
    return coef_list


result = list(map(str, simplified_form(numerator, denominator)))

# 239/30
# 7 1 29
helper.write_stdout(' '.join(result))
