import numpy as np
cimport numpy as np
cimport cython
import numpy

from cython.parallel import prange, threadid
from cython.operator cimport dereference

from cpython.version cimport PY_MAJOR_VERSION

from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free, calloc, rand
from libcpp.string cimport string

from libc.stdint cimport uint32_t, uint64_t 

from cpython.string cimport PyString_AsString

from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release

cimport AVX_cpp as AVX

# Hacky. Should move into pxd file.
cdef extern from "xxhash.c" nogil:
    unsigned XXH32 (const void*, size_t, unsigned)
    unsigned long long XXH64 (const void*, size_t, unsigned long long)

# cdef extern from "<string>" namespace "std" nogil:
#   cdef cppclass string :
#     string()
#     string(const char* s)
#     size_t length()
#     const char* c_str()
#     char& operator[] (size_t)

def preallocate_locks(num_locks) :
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks

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
  unsigned num_questions
  unsigned *num_words_per_question
  unsigned num_threads
  char ***question_texts
  unsigned [:,:] tf_vectors
  float [:,:] tfidf_vectors
  float [:] idf_vector
  uint32_t [:] simhashes32
  uint64_t [:] simhashes64
  unsigned [:, :] distances64
  unsigned [:, :] distances32 

cpdef init_globals(N) :
  global word_indices, num_threads
  word_indices = new unordered_map[string, unsigned]()
  num_threads = N

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

cpdef load_questions(questions) :
  global question_texts, num_questions, num_words_per_question

  # Convert from python object to c pointers
  num_questions = len(questions)

  # Allocate space for pointers to each question
  question_texts = <char ***>malloc(num_questions * sizeof(char **))

  # Allocate lengths per question
  num_words_per_question = <unsigned *>malloc(num_questions * sizeof(unsigned))

  for i in xrange(num_questions) :
    question = questions[i]

    # Allocate space for pointers to each word in question
    question_texts[i] = <char **>malloc(len(question) * sizeof(char *))
    
    # Allocate space for each word in question and copy.
    num_words_per_question[i] = len(question)
    for j in xrange(len(question)) :
      question_texts[i][j] = PyString_AsString(question[j])

cpdef init_tfs(unsigned num_indices) :
  global tf_vectors, num_threads, word_indices
  tf_vectors = np.zeros([num_questions,num_indices]).astype(np.uint32)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef create_tfs() :
  global num_questions, num_threads, num_words_per_question, word_indices, \
          tf_vectors, question_texts
  cdef :
    unordered_map[string, unsigned].iterator iter_value
    unsigned i, j, word_index
  for i in prange(num_questions, 
                  nogil=True, 
                  chunksize=num_questions/8, 
                  num_threads=num_threads, 
                  schedule="static") :
    for j in xrange(num_words_per_question[i]) :
      iter_value = word_indices.find(string(question_texts[i][j]))
      if word_indices.end() != iter_value :
        word_index = dereference(iter_value).second
        tf_vectors[i][word_index] += 1
  return tf_vectors

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_idf(unsigned num_indices, uintptr_t locks_ptr, unsigned num_locks) :
  global num_questions, num_words_per_question, idf_vector, word_indices, \
          question_texts
  
  idf_vector = np.log(num_questions) * np.ones(num_indices).astype(np.float32)

  cdef :
    unsigned [:,:] temp_vectors = np.zeros([num_threads, num_indices]).astype(np.uint32)
    unsigned i, j, word_index, tid
    unsigned [:] num_docs_vector = np.zeros(num_indices).astype(np.uint32)
    omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

  for i in prange(num_questions,
                  nogil=True, 
                  chunksize=num_questions/8, 
                  num_threads=num_threads, 
                  schedule="static") :
    tid = threadid()

    for j in xrange(num_words_per_question[i]) :
      iter_value = word_indices.find(string(question_texts[i][j]))
      if word_indices.end() != iter_value :
        word_index = dereference(iter_value).second
        if temp_vectors[tid, word_index] == 0 :
          temp_vectors[tid, word_index] += 1
          acquire(&locks[word_index%num_locks])
          num_docs_vector[word_index] += 1
          release(&locks[word_index%num_locks])

    for j in xrange(num_indices) :
      temp_vectors[tid, j] = 0
  
  for j in xrange(num_indices) :
    idf_vector[j] -= np.log(num_docs_vector[j])

  return idf_vector

cpdef init_tfidfs(unsigned num_indices) :
  global tfidf_vectors, num_threads, word_indices
  tfidf_vectors = np.copy(tf_vectors).astype(np.float32)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_tfidfs(unsigned num_indices, AVX_f) :
  global tfidf_vectors, idf_vector, tf_vectors
  cdef:
    AVX.float8 tfidf_float8, result_float8
    AVX.float8 *idf_float8s
    unsigned i, j

  if AVX_f :
    assert(num_indices % 8 == 0)
    idf_float8s = <AVX.float8 *>malloc(num_indices / 8 * sizeof(AVX.float8))
    for j in range(0, num_indices, 8) :
        idf_float8s[j/8] = AVX.make_float8(idf_vector[j+7],
                                  idf_vector[j+6],
                                  idf_vector[j+5],
                                  idf_vector[j+4],
                                  idf_vector[j+3],
                                  idf_vector[j+2],
                                  idf_vector[j+1],
                                  idf_vector[j])
    for i in prange(num_questions,
                    nogil=True, 
                    chunksize=num_questions/4, 
                    num_threads=num_threads, 
                    schedule="static") :
      for j in range(0, num_indices, 8) :
        tfidf_float8 = AVX.make_float8(tfidf_vectors[i,j+7],
                                  tfidf_vectors[i,j+6],
                                  tfidf_vectors[i,j+5],
                                  tfidf_vectors[i,j+4],
                                  tfidf_vectors[i,j+3],
                                  tfidf_vectors[i,j+2],
                                  tfidf_vectors[i,j+1],
                                  tfidf_vectors[i,j])
        result_float8 = AVX.mul(idf_float8s[j/8], tfidf_float8)
        AVX.to_mem(result_float8, &(tfidf_vectors[i, j]))
  else :
    for i in prange(num_questions,
                    nogil=True, 
                    chunksize=num_questions/4, 
                    num_threads=num_threads, 
                    schedule="static") :
      for j in range(num_indices) :
        tfidf_vectors[i, j] *= idf_vector[j]
  return tfidf_vectors


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_simhashes64(unsigned size):
  global word_indices, question_texts, simhashes32, simhashes64, num_threads, \
        tfidf_vectors, num_words_per_question
  cdef:
    unsigned i, u, counter, word_index, bit, tid
    float weight
    string word
    uint64_t wordhash64, simhash64
    float [:,:] W = np.zeros([num_threads, size]).astype(np.float32)

  simhashes64 = np.zeros(num_questions).astype(np.uint64)
  for u in prange(num_questions,
                  nogil=True,
                  chunksize=1,
                  num_threads=num_threads,
                  schedule='static'):
    tid = threadid()

    for i in xrange(num_words_per_question[u]):
      word = string(question_texts[u][i])
      iter_value = word_indices.find(word)

      if word_indices.end() != iter_value :
        word_index = dereference(iter_value).second
        weight = tfidf_vectors[u, word_index]
        counter = size-1
        wordhash64 = hash64(word)
        while wordhash64 > 0:
          bit = wordhash64 % 2
          if bit :
            W[tid, counter] += weight
          else :
            W[tid, counter] -= weight
          wordhash64 = wordhash64 >> 1
          counter = counter - 1

    simhash64 = 0
    for i in xrange(size):
      simhash64 = simhash64 << 1
      if W[tid, i] >= 0:
        simhash64 = simhash64 + 1

    simhashes64[u] = simhash64
  return simhashes64

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_simhashes32(unsigned size):
  global word_indices, question_texts, simhashes32, simhashes64, num_threads, \
        tfidf_vectors, num_words_per_question
  cdef:
    unsigned i, u, counter, word_index, bit, tid
    float weight
    string word
    uint32_t wordhash32, simhash32
    float [:,:] W = np.zeros([num_threads, size]).astype(np.float32)

  simhashes32 = np.zeros(num_questions).astype(np.uint32)
  for u in prange(num_questions,
                  nogil=True,
                  chunksize=1,
                  num_threads=num_threads,
                  schedule='static'):
    tid = threadid()

    for i in xrange(num_words_per_question[u]):
      word = string(question_texts[u][i])
      iter_value = word_indices.find(word)

      if word_indices.end() != iter_value :
        word_index = dereference(iter_value).second
        weight = tfidf_vectors[u, word_index]
        counter = size-1
        wordhash32 = hash32(word)
        while wordhash32 > 0:
          bit = wordhash32 % 2
          if bit :
            W[tid, counter] += weight
          else :
            W[tid, counter] -= weight
          wordhash32 = wordhash32 >> 1
          counter = counter - 1

    simhash32 = 0
    for i in xrange(size):
      simhash32 = simhash32 << 1
      if W[tid, i] >= 0:
        simhash32 = simhash32 + 1

    simhashes32[u] = simhash32
  return simhashes32

cdef uint64_t hash64(string str1) nogil:
  return XXH64(<const void*> str1.c_str(), str1.length(), rand())
  
cdef uint32_t hash32(string str1) nogil:
  return XXH32(<const void*> str1.c_str(), str1.length(), rand())

cpdef calculate_distances64(unsigned size):
  global num_questions, simhashes64, num_threads, distances64
  cdef:
    unsigned i, j

  distances64 = np.zeros([num_questions, num_questions]).astype(np.uint32)

  for i in prange(num_questions, nogil=True, chunksize=1, num_threads=num_threads, schedule='static'):
    for j in xrange(num_questions):
      distances64[i, j] = numBits64(simhashes64[i] ^ simhashes64[j])
  return distances64

#The following are from https://yesteapea.wordpress.com/2013/03/03/counting-the-number-of-set-bits-in-an-integer/

cdef unsigned numBits64(uint64_t i) nogil:

  i = i - ((i >> <uint64_t>1) & <uint64_t> 0x5555555555555555)
  i = (i & <uint64_t> 0x3333333333333333) + ((i >> <uint64_t> 2) & <uint64_t> 0x3333333333333333)
  i = ((i + (i >> <uint64_t> 4)) & <uint64_t> 0x0F0F0F0F0F0F0F0F)
  return (i*(<uint64_t> 0x0101010101010101))>> <uint64_t> 56


cpdef calculate_distances32(unsigned size):
  global num_questions, simhashes32, num_threads, distances32
  cdef:
    unsigned i, j

  distances32 = np.zeros([num_questions, num_questions]).astype(np.uint32)

  for i in prange(num_questions, nogil=True, chunksize=1, num_threads=num_threads, schedule='static'):
    for j in xrange(num_questions):
      distances32[i, j] = numBits32(simhashes32[i] ^ simhashes32[j])
  return distances32 

cdef unsigned numBits32(uint64_t i) nogil:
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = ((i + (i >> 4)) & 0x0F0F0F0F)
    return (i*(0x01010101))>>24
