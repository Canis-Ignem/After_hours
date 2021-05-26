from no_main import last
import numpy as np


print("hey")

def five():
    print("this is the last action",np.array([5]))

def four():
    print("this is the last action",np.array([4]))

def three():
    print("this is the last action",np.array([3]))

def nevercalled():
    print("AAAAAAAAAAAAAAAAAAAAAAAAA")
    
if __name__ == '__main__':
    three()
    four()
    five()
else:
    print('This file was imported by another')