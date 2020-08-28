from hashlib import md5
from time import localtime

def get_unique_filename(filename):
    prefix = md5(str(localtime()).encode('utf-8')).hexdigest()
    return f"{prefix}_{filename}"

