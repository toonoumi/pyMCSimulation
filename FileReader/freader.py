
import logging
import struct
import numpy as np

ram_limit=4294967296  #4gb
ram_limit=1024

def mFmt(dtype):
    switcher={
        'float32':'f',
        'float64':'d',
    }
    
    return switcher.get(dtype)

def mSize(dtype):
    switcher={
        'float32':4,
        'float64':8,
    }
    
    return switcher.get(dtype)

def gen_rand_bin_file(fname,size=4096,dtype='float32',min_max=(0,1)):
    fname=fname.strip()
    if fname=='':
        logging.error('Output File name missing.')
    
    single_size=mSize(dtype)    
        
    lst_length=int(size/single_size)  #calculate how many numbers are needed
    single_write_len=lst_length
    it=1
    if size>ram_limit:
        it=size/ram_limit
        single_write_len=int(ram_limit/single_size)
        if not isinstance(it, int):
            it=int(it+1)
    
    f=open(fname,"wb")
    for _ in range(0,it):
        
        lst = np.random.uniform(low=min_max[0], high=min_max[1], size=(single_write_len,))
        lst = np.array(lst,dtype=dtype)
        #print(lst)
        content=bytes(lst)
        #print(content)
        f.write(content)
        lst_length-=single_write_len
        if lst_length<single_write_len:
            single_write_len=lst_length
    f.close()



def fopen(fname):
    '''
    @return the fd of the opened file
    '''
    return open(fname,'rb')

def fclose(fd):
    fd.close()
    
def read_bytes(fd,offset=0,size=1024,dtype='float32'):
    content=fd.read(size)
    fmt=mFmt(dtype)*int(size/mSize(dtype))
    arr=struct.unpack(fmt,content)
    return arr


if __name__ == "__main__":
    #gen_rand_bin_file('test.bin',size=4294967296)
    #fd=fopen("test.bin")
    #rst=read_bytes(fd)
    #print(rst)
    #fclose(fd)
    exit()