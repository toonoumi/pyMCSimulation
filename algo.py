from FileReader import freader as fr 
import time
from chronometer import Chronometer
import numpy as np





if __name__ == "__main__":
    fd= fr.fopen('test.bin')
    
        
    lst=fr.read_bytes(fd,size=429496400)
        
    with Chronometer() as t:
        rst=np.mean(lst)
        
    print(rst," Time consumed: ",t)
    fr.fclose(fd)
    
    0