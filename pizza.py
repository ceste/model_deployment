# import argparse

# parser = argparse.ArgumentParser()

# # Add long and short argument
# parser.add_argument("--user", "-u", help="set user id for tracking", required=True)

# args = parser.parse_args()

# print(args)



# print(self.__class__.__name__)
# log.print('self.__class__.__name__')

# import sys

# def get_function_caller():
# 	print(sys._getframe(1).f_code.co_name)

# def b():
# 	get_function_caller()

# b()


import inspect
def f1(): 
	f2()

def f2():
	curframe = inspect.currentframe()
	calframe = inspect.getouterframes(curframe, 2)
	print('caller name:', calframe[1][3])

f1()
