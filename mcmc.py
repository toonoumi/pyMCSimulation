import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FileReader import freader as fr
import time
from chronometer import Chronometer
from numba import vectorize
import secrets
import struct
import numba
import numpy as np

__MAX_ONETIME_IT_COUNT=200000000
__PRINT_MIDDLE_RST=True

@numba.njit
def func_eg(argv):
    '''
    Example on how to construct your function. 
    @param argv is list of variable for the function.
    @return your function should return its result.
    '''
    return argv[0]**2+argv[1]

@numba.njit
def __mc_gen_rand_sample(func,points,it_count,argc,A,B,rst_range):
    for i in range(it_count):
        for j in range(argc):
            points[i][j]=np.random.rand()*(B[j]-A[j])+A[j]
        points[i][argc]=func(points[i])
        

#@numba.njit
def __mc_test_sample(points,it_count,argc):
    rst=np.mean(points.transpose()[argc],dtype='float64')
    return rst

def __mc_integration_helper(func,argc,A,B,it_count=10000,dtype='float32'):
    '''
        Calculate the integration of a function from array A to array B
        for A=[a1,a2,a3...], B=[b1,b2,b3...], length of A and B must match.
        @param func is the function to run
        @param argc is the number of parameter of such function
        @param it_count is the number of iteration
    '''
    if len(A)!=len(B):
        print("ERROR: length of A and B does not match.")
        return
    if len(A)<argc:
        print('ERROR: param not sufficient.')
    points=np.zeros((it_count,argc+1),dtype=dtype)
    
    #get min max range
    rst_min=func(A)
    rst_max=func(B)
    #generate random sample
    __mc_gen_rand_sample(func,points,it_count,argc,A,B,np.array([rst_min,rst_max]))
    #test random sample
    rst=__mc_test_sample(points,it_count,argc)
    for i in range(argc):
        rst*=(B[i]-A[i])
    return rst

def mc_integration(func,argc,A,B,it_count=10000,dtype='float32'):
    '''
        Calculate the integration of a function from array A to array B
        for A=[a1,a2,a3...], B=[b1,b2,b3...], length of A and B must match.
        if a large number is used, please use float64 instead of float32
        @param func is the function to run
        @param argc is the number of parameter of such function
        @param it_count is the number of iteration
    '''
    if(it_count<=__MAX_ONETIME_IT_COUNT):
        return __mc_integration_helper(func,argc,A,B,it_count,dtype)
    remain=it_count
    rst=[]
    if __PRINT_MIDDLE_RST:
        print('Number of sampling exceeded maximum, check the setting parameters to adjust.')
        print('Execution is devided into %d runs.' % (remain/__MAX_ONETIME_IT_COUNT+1))
    while remain>0:
        if(remain>__MAX_ONETIME_IT_COUNT):
            rst.append(__mc_integration_helper(func,argc,A,B,__MAX_ONETIME_IT_COUNT,dtype))
            remain-=__MAX_ONETIME_IT_COUNT
        else:
            rst.append(__mc_integration_helper(func,argc,A,B,remain,dtype))
            remain=0
        if __PRINT_MIDDLE_RST:
                print(len(rst)," run, result:",rst[len(rst)-1])
    return np.mean(rst)


if __name__ == "__main__":
    '''
    fd= fr.fopen('test.bin')   
    lst=fr.read_bytes(fd,size=429496400)       
    with Chronometer() as t:
        rst=np.mean(lst)      
    print(rst," Time consumed: ",t)
    fr.fclose(fd)
    '''
    with Chronometer() as t:
        rst=mc_integration(func_eg,2,np.array([2000000000,40000],dtype='float32'),np.array([2000000001,40001],dtype='float64'),it_count=2*10**8)
    print(rst," Time consumed: ",t)
    0