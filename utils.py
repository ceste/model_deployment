from hashlib import md5
from time import localtime
import sys
import inspect

def get_unique_filename(filename):
    prefix = md5(str(localtime()).encode('utf-8')).hexdigest()
    return f"{prefix}_{filename}"

# def get_function_caller():
#     return sys._getframe(1).f_code.co_name




# def f1(): 
#     f2()

def get_function_caller():
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    return calframe[1][3]

# f1()
