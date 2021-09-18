import os
import time
from collections import Iterable
from pprint import pprint


# Timer for supervising function
def timer(func):
    def calc_time(*args):
        start_time = time.time()
        res = func(*args)
        end_time = time.time()
        if isinstance(res, Iterable) and not isinstance(res, str):
            return (*res, end_time - start_time)
        else:
            return res, end_time - start_time

    return calc_time


def check_file(*files: str):
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(
                '{} not exist, please confirm the presence of the target!'.format(file))


def color_print(content: str):
    print(f'\033[0;30;43m{content}\033[0m')


def sep_print(obj: any, start=True, end=True, desc=None, num=80):
    if start:
        print('-' * num)
    if desc:
        print(desc + ':')
    if isinstance(obj, str):
        print(obj)
    else:
        pprint(obj)
    if end:
        print('-' * num)
