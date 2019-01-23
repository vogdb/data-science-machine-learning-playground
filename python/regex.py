import re


def r_usage():
    def print_match(m):
        if m:
            print(m)
            print(m.group(0))
        else:
            print('No match')

    def backslash():
        print('backslash')
        pattern = r'\\'
        p = re.compile(pattern)
        m = p.search(r'\\')
        print_match(m)

    def newline():
        print('newline')
        m = re.search(r'\n', r'\n')
        print_match(m)
        m = re.search(r'\n', '\n')
        print_match(m)
        m = re.search('\n', r'\n')
        print_match(m)
        m = re.search('\n', '\n')
        print_match(m)

    backslash()
    newline()


def multiline_named_groups():
    pattern = r'''
    ^                           # beginning of string
    (                           # beginning of the first half
        (                       # either `first` and `second` together
            first-(?P<first>\w+) 
            _                         
            second-(?P<second>\w+)    
        )|(?P<third>[\w-]+)     # or `third` separate
    )                           # end of the first half
    _-_                       #
    square_(?P<square>[\w.]+) # `square` part
    _-_                       #
    type_(?P<type>[\d]+)      # `type` part
    $                         # end of string
    '''
    p = re.compile(pattern, re.VERBOSE)  # | re.DEBUG

    m = p.search('first-tkb01a1_ch4_cc1_second-t01a2_cc2_h_z_1_-_square_x1.050_z1.00_-_type_37')
    assert m.group('first') == 'tkb01a1_ch4_cc1'
    assert m.group('second') == 't01a2_cc2_h_z_1'
    assert m.group('square') == 'x1.050_z1.00'
    assert m.group('type') == '37'

    m = p.search('C0508456A-I_-_square_g.000_y50_z1.000_-_type_02')
    assert m.group('first') is None
    assert m.group('second') is None
    assert m.group('third') == 'C0508456A-I'
    assert m.group('square') == 'g.000_y50_z1.000'
    assert m.group('type') == '02'


multiline_named_groups()
