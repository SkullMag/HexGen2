"""Query the loaded NCCL API, independently of Torch's compile-time headers."""
import ctypes,os
from pathlib import Path

def mapped_libraries():
    maps=Path('/proc/self/maps')
    return sorted({line.split()[-1] for line in maps.read_text().splitlines() if 'libnccl.so' in line}) if maps.exists() else []

def runtime_version():
    try:fn=ctypes.CDLL(None).ncclGetVersion
    except AttributeError:
        libraries=mapped_libraries()
        if len(libraries)!=1:raise RuntimeError('Cannot identify one loaded NCCL library')
        fn=ctypes.CDLL(libraries[0],mode=os.RTLD_NOLOAD).ncclGetVersion
    fn.argtypes=[ctypes.POINTER(ctypes.c_int)];fn.restype=ctypes.c_int
    code=ctypes.c_int();status=fn(ctypes.byref(code))
    if status!=0:raise RuntimeError('ncclGetVersion failed: '+str(status))
    return code.value//10000,(code.value%10000)//100,code.value%100
