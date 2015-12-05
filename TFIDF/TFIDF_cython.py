import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

# import tfidf_dictionary as tfidf
import numpy as np

import tfidf_cython_serial as tfidf

import mandelbrot

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

# def get_value_at_key(key) :
#   return tfidf.get_index(key)
    
# def calculate_tfidfs() :
#   global question_texts, word_indices, tfidf_vectors, tfidf_norms, idf_vector
#   num_docs_vector = np.zeros(len(word_indices))
#   N = float(len(question_texts))
#   for text in question_texts :
#       temp_dict = {}
#       for word in text :
#         if word in word_indices :
#           if word in temp_dict :
#             continue
#           else :
#             temp_dict[word] = True
#             num_docs_vector[word_indices[word]] += 1
#   idf_vector = np.log(N / num_docs_vector)
#   tfidf_vectors = tf_vectors * idf_vector[None, :]
#   tfidf_norms = np.linalg.norm(tfidf_vectors, axis=1)

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
