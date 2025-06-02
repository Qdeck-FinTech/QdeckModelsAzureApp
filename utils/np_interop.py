import numpy as np
import ctypes
import clr, System
from System import Array, Int32, IntPtr
from System.Runtime.InteropServices import GCHandle, GCHandleType, Marshal
from ctypes import string_at


_MAP_NP_NET = {
    np.dtype("float32"): System.Single,
    np.dtype("float64"): System.Double,
    np.dtype("int8"): System.SByte,
    np.dtype("int16"): System.Int16,
    np.dtype("int32"): System.Int32,
    np.dtype("int64"): System.Int64,
    np.dtype("uint8"): System.Byte,
    np.dtype("uint16"): System.UInt16,
    np.dtype("uint32"): System.UInt32,
    np.dtype("uint64"): System.UInt64,
    np.dtype("bool"): System.Boolean,
}

_MAP_NET_NP = {
    "Single": np.dtype("float32"),
    "Double": np.dtype("float64"),
    "SByte": np.dtype("int8"),
    "Int16": np.dtype("int16"),
    "Int32": np.dtype("int32"),
    "Int64": np.dtype("int64"),
    "Byte": np.dtype("uint8"),
    "UInt16": np.dtype("uint16"),
    "UInt32": np.dtype("uint32"),
    "UInt64": np.dtype("uint64"),
    "Boolean": np.dtype("bool"),
}


class DictToObj:
    def __init__(self, in_dict: dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(
                    self, key, [DictToObj(x) if isinstance(x, dict) else x for x in val]
                )
            else:
                setattr(self, key, DictToObj(val) if isinstance(val, dict) else val)


def marshal_float_to_numpy(netArray):
    """
    Given a CLR `System.Double[]` returns a `numpy.ndarray`
    """
    dims = np.empty(netArray.Rank, dtype=int)
    for I in range(netArray.Rank):
        dims[I] = netArray.GetLength(I)

    npArray = np.empty(dims, order="C", dtype=np.float64)
    destPtr = npArray.__array_interface__["data"][0]
    Marshal.Copy(netArray, 0, destPtr, len(netArray))

    return npArray


def buffer_float_to_numpy(netArray):
    """
    Given a CLR `System.Double[]` returns a `numpy.ndarray` of np.float64
    """
    try:
        sourceHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = sourceHandle.AddrOfPinnedObject().ToInt64()
        npArray = np.frombuffer(
            string_at(sourcePtr, len(netArray) * 8), dtype=np.float64
        )
    finally:
        if sourceHandle.IsAllocated:
            sourceHandle.Free()

    return npArray


def to_numpy(netArray):
    """
    Given a CLR `System.Array` returns a `numpy.ndarray`.  See _MAP_NET_NP for
    the mapping of CLR types to Numpy dtypes.
    """
    dims = np.empty(netArray.Rank, dtype=int)
    for I in range(netArray.Rank):
        dims[I] = netArray.GetLength(I)
    netType = netArray.GetType().GetElementType().Name

    try:
        npArray = np.empty(dims, order="C", dtype=_MAP_NET_NP[netType])
    except KeyError:
        raise NotImplementedError(
            "to_numpy does not yet support System type {}".format(netType)
        )

    try:  # Memmove
        sourceHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = sourceHandle.AddrOfPinnedObject().ToInt64()
        destPtr = npArray.__array_interface__["data"][0]
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if sourceHandle.IsAllocated:
            sourceHandle.Free()
    return npArray


def to_netarray(npArray):
    """
    Given a `numpy.ndarray` returns a CLR `System.Array`.  See _MAP_NP_NET for
    the mapping of Numpy dtypes to CLR types.

    Note: `complex64` and `complex128` arrays are converted to `float32`
    and `float64` arrays respectively with shape [m,n,...] -> [m,n,...,2]
    """
    dims = npArray.shape
    dtype = npArray.dtype
    # For complex arrays, we must make a view of the array as its corresponding
    # float type.
    if dtype == np.complex64:
        dtype = np.dtype("float32")
        dims.append(2)
        npArray = npArray.view(np.float32).reshape(dims)
    elif dtype == np.complex128:
        dtype = np.dtype("float64")
        dims.append(2)
        npArray = npArray.view(np.float64).reshape(dims)

    if not npArray.flags.c_contiguous:
        npArray = npArray.copy(order="C")
    assert npArray.flags.c_contiguous

    try:
        netArray = Array.CreateInstance(_MAP_NP_NET[dtype], *npArray.shape)
    except KeyError:
        raise NotImplementedError(
            "to_netarray does not yet support dtype {}".format(dtype)
        )

    try:  # Memmove
        destHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = npArray.__array_interface__["data"][0]
        destPtr = destHandle.AddrOfPinnedObject().ToInt64()
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if destHandle.IsAllocated:
            destHandle.Free()
    return netArray


if __name__ == "__main__":

    from time import perf_counter
    import psutil

    tries = 1000
    foo = np.full([1024, 1024], 2.5, dtype="float32")

    netMem = np.zeros(tries)
    t_asNet = np.zeros(tries)
    netFoo = to_netarray(foo)  # Lazy loading makes the first iteration very slow
    for I in range(tries):
        t0 = perf_counter()
        netFoo = to_netarray(foo)
        t_asNet[I] = perf_counter() - t0
        netMem[I] = psutil.virtual_memory().free / 2.0**20

    t_asNumpy = np.zeros(tries)
    numpyMem = np.zeros(tries)
    unNetFoo = to_numpy(netFoo)  # Lazy loading makes the first iteration very slow
    for I in range(tries):
        t0 = perf_counter()
        unNetFoo = to_numpy(netFoo)
        t_asNumpy[I] = perf_counter() - t0
        numpyMem[I] = psutil.virtual_memory().free / 2.0**20

    # Convert times to milliseconds
    t_asNet *= 1000
    t_asNumpy *= 1000
    np.testing.assert_array_almost_equal(unNetFoo, foo)
    print(
        "Numpy to .NET converted {} bytes in {:.3f} +/- {:.3f} ms (mean: {:.1f} ns/ele)".format(
            foo.nbytes, t_asNet.mean(), t_asNet.std(), t_asNet.mean() / foo.size * 1e6
        )
    )
    print(
        ".NET to Numpy converted {} bytes in {:.3f} +/- {:.3f} ms (mean: {:.1f} ns/ele)".format(
            foo.nbytes,
            t_asNumpy.mean(),
            t_asNumpy.std(),
            t_asNumpy.mean() / foo.size * 1e6,
        )
    )

    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.plot(np.arange(tries), netMem, '-', label='to_netarray')
    # plt.plot(np.arange(tries), numpyMem, '-', label='to_numpy')
    # plt.legend(loc='best')
    # plt.ylabel('Free memory (MB)')
    # plt.xlabel('Iteration')
    # plt.show(block=True)
