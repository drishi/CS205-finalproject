import sys
import os.path
sys.path.append(os.path.join('', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np

import tfidf_cython_serial as tfidf

from timer import Timer

def print_t(t, string) :
  print "Time for " + string + ": " + str(t.interval)

def init_globals(N) :
  '''
  Load questions into memory
  '''
  with Timer() as t :
    tfidf.init_globals(N)
  print_t(t, "Initialization")

def load_questions(questions) :
  '''
  Load questions into memory
  '''
  with Timer() as t :
    tfidf.load_questions(np.array(questions))
  print_t(t, "load_questions")

def load_indices(indices) :
  '''
  Load word indices into memory
  '''
  global num_keys 
  num_keys = len(indices)
  with Timer() as t :
    for (key, value) in indices :
      tfidf.load_index(key.encode('utf-8', 'ignore'), value)
  print_t(t, "load_indices")

def init_tfs() :
  '''
  Create Term Frequencies for questions
  '''
  global num_keys
  with Timer() as t :
    tfidf.init_tfs(num_keys)
  print_t(t, "init_tfs")

def create_tfs() :
  '''
  Create Term Frequencies for questions
  '''
  global num_keys
  with Timer() as t :
    tfs = tfidf.create_tfs()
  print_t(t, "create_tfs")
  return tfs

def calculate_idf(num_locks) :
  global num_keys
  with Timer() as t:
    locks_ptr = tfidf.preallocate_locks(num_locks)
    idf = tfidf.calculate_idf(num_keys, locks_ptr, num_locks)
  print_t(t, "calculate_idf")
  return idf

def init_tfidfs() :
  global num_keys
  with Timer() as t:
    tfidfs = tfidf.init_tfidfs(num_keys)
  print_t(t, "int_tfidfs")

def calculate_tfidfs() :
  global num_keys
  with Timer() as t:
    tfidfs = tfidf.calculate_tfidfs(num_keys)
  print_t(t, "calculate_tfidfs")
  return tfidfs

# # def calculate_tfidf(example_question) :
# #   global word_indices
# #   tf_vector = np.zeros(len(word_indices))
# #   for word in example_question :
# #     if word in word_indices :
# #       tf_vector[word_indices[word]] += 1
# #   return tf_vector * idf_vector

# # def calculate_cossim(example_question) :
# #   global tfidf_vectors, tfidf_norms
# #   example = calculate_tfidf(example_question)
# #   return tfidf_vectors.dot(example) / (np.linalg.norm(example) * tfidf_norms)
