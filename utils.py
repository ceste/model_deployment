from compress_pickle import dump, load
import io
from hashlib import md5
from time import localtime
import sys
import inspect
import config

def get_unique_filename(filename):
    prefix = md5(str(localtime()).encode('utf-8')).hexdigest()
    return f"{prefix}_{filename}"

def get_function_caller():
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    return calframe[1][3]

def compress_files(filename, obj):
    fname = str(filename)
    dump(obj, fname, compression="lzma", set_default_extension=False)

def load_compressed_files(filename):    
    fname = str(filename)
    return load(fname, compression="lzma", set_default_extension=False)

