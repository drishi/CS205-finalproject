import numpy as np
cimport numpy as np
cimport cython
import numpy

from cython.parallel import prange
from cython.operator cimport dereference

from cpython.version cimport PY_MAJOR_VERSION

from libc.stdlib cimport malloc, free
from libcpp.string cimport string

from cpython.string cimport PyString_AsString

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
  global tf_vectors, question_texts, num_threads, word_indices
  tf_vectors = np.zeros([num_questions,num_indices]).astype(np.uint32)
  print "Values Initialized"

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef create_tfs() :
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
      tf_vectors[i][5] += 1
  return tf_vectors


cpdef calculate_tfidfs() :
  global question_texts, 

