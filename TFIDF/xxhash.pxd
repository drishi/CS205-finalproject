# Cython Definition file for AVX.h

cdef extern from "xxhash.h" nogil:

    unsigned XXH32 (const void*, size_t, unsigned)

    unsigned long long XXH64 (const void*, size_t, unsigned long long)
