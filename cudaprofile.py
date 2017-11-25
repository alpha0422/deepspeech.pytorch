import ctypes

#_cudart = ctypes.CDLL('libcudart.so')
#_cudart = ctypes.CDLL('/usr/local/cuda/r80/lib64/libcudart.so')
_cudart = ctypes.CDLL('/home/scratch.wkong_gpu/toolbox/anaconda2/lib/python2.7/site-packages/torch/lib/libcudart.so')


def start():
    # As shown at http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html,
    # the return value will unconditionally be 0. This check is just in case it changes in
    # the future.
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)

def stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)

