'''
Numpy serial implemnation of TFIDF
'''
import numpy as np
import sys
import os.path
sys.path.append(os.path.join('..', 'util'))
from timer import Timer

import xxhash

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
      word_indices[key.encode('utf-8', 'ignore')] = value
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
          word = word.encode('utf-8', 'ignore')
          if word in word_indices :
            if word in temp_dict :
              continue
            else :
              temp_dict[word] = True
              num_docs_vector[word_indices[word]] += 1
    idf_vector = np.log(float(N)) - np.log(num_docs_vector)
  print_t(t, "create_idf")
  return idf_vector

def calculate_tfidfs() :  
  global tfidf_vectors
  with Timer() as t :
    tfidf_vectors = tf_vectors * idf_vector[None, :]
  print_t(t, "calculate_tfidfs")
  return tfidf_vectors

def calculate_tfidf_norms() :
  global tfidf_norms
  with Timer() as t :
    tfidf_norms = np.linalg.norm(tfidf_vectors, axis=1)
  print_t(t, "calculate_tfidf_norms")

def calculate_tfidf(example_question) :
  global word_indices, idf_vector
  with Timer() as t :
    tf_vector = np.zeros(len(word_indices))
    for word in example_question :
      if word in word_indices :
        tf_vector[word_indices[word]] += 1
  print_t(t, "calculate_tfidf")
  return tf_vector * idf_vector

def calculate_cossims() :
  global tfidf_vectors
  with Timer() as t :
    norms = np.sum(np.abs(tfidf_vectors)**2,axis=-1)**(1./2)
    I = norms > 0
    u_vectors = np.zeros(tfidf_vectors.shape)
    u_vectors[I] = tfidf_vectors[I] / norms[I][:, None]
    cossims = u_vectors.dot(u_vectors.T)
  print_t(t, "calculate_cossims")
  return cossims

def calculate_simhashes() :
  global word_indices, question_texts, tfidf_vectors, simhashes
  with Timer() as t :
    simhashes = []
    for u in range(len(question_texts)):
      W = np.zeros(64)
      for i in range(len(question_texts[u])):
        word = question_texts[u][i]
        if word not in word_indices :
          continue
        wordhash = xxhash.xxh64(word).intdigest()
        counter = 63
        while wordhash > 0:
            bit = wordhash % 2
            if bit:
                W[counter] += tfidf_vectors[u][word_indices[word]]
            else:
                W[counter] -= tfidf_vectors[u][word_indices[word]]
            wordhash = wordhash >> 1
            counter -= 1
      simhash = 0
      for i in range(len(W)):
          if W[i] >= 0:
              simhash += 1
          if i < len(W)-1:
              simhash = simhash << 1
      simhashes.append(simhash)
  print_t(t, "calculate_simhashes")
  return simhashes

def calculate_distances() :
  global simhashes
  with Timer() as t :
    A = np.array([simhashes] * len(simhashes))
    distances = numBits64(A ^ A.T)
  print_t(t, "calculate_distances")
  return distances

def numBits64(i):
    i = i - ((i >> np.uint64(1)) & np.uint64(0x5555555555555555))
    i = (i & np.uint64(0x3333333333333333)) + ((i >> np.uint64(2)) & np.uint64(0x3333333333333333))
    i = ((i + (i >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F))
    return (i*(np.uint64(0x0101010101010101)))>>np.uint64(56)  
