from FileReader import freader as fr 
import time
from chronometer import Chronometer
from numba import vectorize
import numba
import numpy as np

@numba.njit
def square(x):
    return x[0]**2

@numba.njit
def __mc_gen_rand_sample(func,points,it_count,argc,A,B,rst_range):
    for i in range(it_count):
        for j in range(argc):
            #points[i][j]=np.random.uniform(low=A[j], high=B[j])
            points[i][j]=np.random.rand()*(B[j]-A[j])+A[j]
        #points[i][argc]=np.random.uniform(low=rst_range[0],high=rst_range[1])
        points[i][argc]=func(points[i])
        

@numba.njit
def __mc_test_sample(points,it_count,argc):
    hit=0
    for i in range(it_count):
       hit+=points[i][argc]
    return hit/it_count

def mc_integration(func,argc,A,B,it_count=100000,dtype='float32'):
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
    #print(rst_min," ",rst_max)
    #generate random sample
    __mc_gen_rand_sample(func,points,it_count,argc,A,B,np.array([rst_min,rst_max]))
    #test random sample
    rst=__mc_test_sample(points,it_count,argc)
    rst*=B[0]-A[0]
    return rst

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
        rst=mc_integration(square,1,np.array([0],dtype='float32'),np.array([3],dtype='float32'),it_count=100000000)
    print(rst," Time consumed: ",t)
    0