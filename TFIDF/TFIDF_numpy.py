'''
Numpy serial implemnation of TFIDF
'''
import numpy as np

import sys
import os.path
sys.path.append(os.path.join('..', 'util'))
from timer import Timer

def print_t(t, string) :
  print "Time for " + string + ": " + str(t.interval)

def init_globals() :
  global tf_vectors, question_texts, word_indices, \
            tfidf_vectors, tfidf_norms, idf_vector  
  with Timer() as t :  
    tf_vectors = None
    question_texts = None
    word_indices = None
    tfidf_vectors = None
    tfidf_norms = None
    idf_vector = None  
  print_t(t, "initialization")

def load_questions(questions) :
  '''
  Load questions into memory
  '''
  global question_texts
  with Timer() as t :
    question_texts = questions
  print_t(t, "load_questions")

def load_indices(indices) :
  '''
  Load word indices into memory
  '''
  global word_indices
  with Timer() as t :
    word_indices = {}
    for (key, value) in indices :
      word_indices[key] = value
  print_t(t, "load_indices")

def init_tfs() :
  with Timer() as t :
    global tf_vectors, question_texts, word_indices
    tf_vectors = np.zeros([len(question_texts), len(word_indices)]).astype(int)
  print_t(t, "init_tfs")

def create_tfs() :
  '''
  Create Term Frequencies for questions
  '''
  global tf_vectors, question_texts, word_indices
  with Timer() as t :
    for i in xrange(len(question_texts)) :
      for word in question_texts[i] :
        if word in word_indices :
          tf_vectors[i][word_indices[word]] += 1
  print_t(t, "create_tfs")
  return tf_vectors

def calculate_idf() :
  global question_texts, word_indices, tfidf_norms, idf_vector
  with Timer() as t :
    num_docs_vector = np.zeros(len(word_indices))
    N = float(len(question_texts))
    for text in question_texts :
        temp_dict = {}
        for word in text :
          if word in word_indices :
            if word in temp_dict :
              continue
            else :
              temp_dict[word] = True
              num_docs_vector[word_indices[word]] += 1
    idf_vector = np.log(N / num_docs_vector)
  print_t(t, "create_idf")

def calculate_tfidfs() :  
  global tfidf_vectors, tfidf_norms
  with Timer() as t :
    tfidf_vectors = tf_vectors * idf_vector[None, :]
    tfidf_norms = np.linalg.norm(tfidf_vectors, axis=1)
  print_t(t, "calculate_tfidfs")

def calculate_tfidf(example_question) :
  global word_indices, idf_vector
  with Timer() as t :
    tf_vector = np.zeros(len(word_indices))
    for word in example_question :
      if word in word_indices :
        tf_vector[word_indices[word]] += 1
  print_t(t, "calculate_tfidf")
  return tf_vector * idf_vector

def calculate_cossim(example_question) :
  global tfidf_vectors, tfidf_norms
  with Timer() as t :
    example = calculate_tfidf(example_question)
    result = tfidf_vectors.dot(example) / (np.linalg.norm(example) * tfidf_norms)
  print_t(t, "calculate_cossim")
  return result
