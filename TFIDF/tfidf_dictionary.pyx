import numpy as np
cimport numpy as np
cimport cython
cimport AVX
import numpy

from cython.parallel cimport prange
from cython.operator cimport dereference

from libcpp.string cimport string

from cpython.version cimport PY_MAJOR_VERSION

cdef unicode _ustring(s):
    if type(s) is unicode:
        # fast path for most common case(s)
        return <unicode>s
    elif PY_MAJOR_VERSION < 3 and isinstance(s, bytes):
        # only accept byte strings in Python 2.x, not in Py3
        return (<bytes>s).decode('ascii', 'ignore')
    elif isinstance(s, unicode):
        # an evil cast to <unicode> might work here in some(!) cases,
        # depending on what the further processing does.  to be safe,
        # we can always create a copy instead
        return unicode(s)
    else:
        raise TypeError(...)

cdef extern from "<unordered_map>" namespace "std" nogil:
    cdef cppclass unordered_map[K, T]: # K: key_type, T: mapped_type
        cppclass pair :
            T& second
        cppclass iterator:
            pair& operator*()
            bint operator==(iterator)
            bint operator!=(iterator)
        unordered_map()
        bint empty()
        size_t size()
        iterator begin()
        iterator end()
        pair emplace(K, T)
        iterator find(K)
        void clear()
        size_t count(K)
        T& operator[](K)
        T& at(const K)

cdef:
  unordered_map[string, unsigned] *word_indices

cpdef init_globals() :
  global word_indices
  with nogil :
    word_indices = new unordered_map[string, unsigned]()

cpdef load_index(string py_key, 
                 int value) :
  global word_indices
  cdef:
    string key = _ustring(py_key)
  with nogil :
    word_indices.emplace(key,value)

cpdef get_index(string py_key) :
  global word_indices
  cdef:
    unordered_map[string, unsigned].iterator i
    string key = _ustring(py_key)
    int index = -1
  
  with nogil :
    i = word_indices.find(key)
    if word_indices.end() != i :
      index = dereference(i).second
  return index