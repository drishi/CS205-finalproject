import numpy as np
cimport numpy as np
cimport cython
import numpy

from cython.parallel import prange, threadid
from cython.operator cimport dereference

from cpython.version cimport PY_MAJOR_VERSION

from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free, calloc
from libcpp.string cimport string

from cpython.string cimport PyString_AsString

from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release

cimport AVX_cpp as AVX

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
cpdef calculate_idf(unsigned num_indices, uintptr_t locks_ptr) :
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
          acquire(&locks[0])
          num_docs_vector[word_index] += 1
          release(&locks[0])

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
    print "Using AVX"
    assert(num_indices % 8 == 0)
    idf_float8s = <AVX.float8 *>malloc(num_indices / 8 * sizeof(AVX.float8))
    for j in range(0, num_indices, 8) :
        idf_float8s[j/8] = AVX.make_float8(tfidf_vectors[i,j+7],
                                  tfidf_vectors[i,j+6],
                                  tfidf_vectors[i,j+5],
                                  tfidf_vectors[i,j+4],
                                  tfidf_vectors[i,j+3],
                                  tfidf_vectors[i,j+2],
                                  tfidf_vectors[i,j+1],
                                  tfidf_vectors[i,j])

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




