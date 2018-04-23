import sys


def read_stdin():
    """return stdin as a list of strings"""
    lines = sys.stdin.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n', '')
    return lines


def read_file(filename):
    """return file as a list of strings"""
    with open(filename) as file:
        lines = file.readlines()
    lines = [line.replace('\n', '') for line in lines]
    return lines


def write_stdout(buffer):
    """writes to stdout"""
    sys.stdout.write(buffer)
