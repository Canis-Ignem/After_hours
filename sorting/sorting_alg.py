from time import time
import numpy as np
simple_list = [5,1,3,8,4,7,2,9,0,6]
np.random.seed(0)
big_list = np.random.permutation(100000)
first = simple_list[0]

def strive_sort(l):
    n = len(l)
    for i in range(n):
        for j in range(n):
            if l[i] <= l[j]:
                
                l[i], l[j] = l[j], l[i]
            
    return l

        
def buble_sort(l):  
    n = len(l)       
    for i in range(n):
        for j in range(0, n-i-1):
            if l[j+1] < l[j]:
                
                l[j], l[j+1] = l[j+1], l[j]
    return l

def partition( l, low, high):
    
    i = low -1
    pivot = l[high]
    
    for j in range( low, high):
        
        if l[j] < pivot:
            i += 1
            l[i], l[j] = l[j], l[i]   
    l[i+1], l[high] = l[high], l[i+1]
    return i+1
        
def quick_sort(left, right, l):
    
    if left < right:
        pivot = partition(l,left, right)
        quick_sort( left, pivot-1, l)
        quick_sort( pivot+1, right, l)

        
    

'''
start = time()
strive_sort(big_list)
end = time()

print("Strive took: ", end-start, "s")

start = time()
buble_sort(big_list)
end = time()

print("buble took: ", end-start, "s")
'''

start = time()
quick_sort(0, len(simple_list)-1,simple_list)
print(simple_list)
end = time()

print("quick took: ", end-start, "s")
