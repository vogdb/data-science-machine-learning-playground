import helper
import re

lines = helper.read_file('5.10_test.txt')

# this task could be interestingly solved with Graphs
conversion_list = [
    {'from': 'mile', 'to': 'm', 'factor': 1609},
    {'from': 'yard', 'to': 'm', 'factor': 0.9144},
    {'from': 'foot', 'to': 'cm', 'factor': 30.48},
    {'from': 'inch', 'to': 'cm', 'factor': 2.54},
    {'from': 'km', 'to': 'm', 'factor': 1000},
    {'from': 'cm', 'to': 'm', 'factor': 0.01},
    {'from': 'mm', 'to': 'm', 'factor': 0.001},
]

conversion_to_meter = {'m': 1}


def to_meter_base():
    # first fill all direct conversions
    for item in conversion_list:
        if (item['from'] != 'm') and (item['to'] == 'm'):
            conversion_to_meter[item['from']] = item['factor']
    # fill indirect conversions to meter
    for item in conversion_list:
        if (item['to'] != 'm') and (item['from'] not in conversion_to_meter):
            conversion_to_meter[item['from']] = item['factor'] * conversion_to_meter[item['to']]


to_meter_base()


# <number> <unit_from> in <unit_to>
matches = re.search('([\de.]+)\s(\w{,5})\sin\s(\w{,5})[\s]*', lines[0])
if matches:
    number, unit_from, unit_to = matches.groups()
    number = float(number)
    number_m = number * conversion_to_meter[unit_from]
    number_to = number_m * (1. / conversion_to_meter[unit_to])
    helper.write_stdout('{:.2E}'.format(number_to))
