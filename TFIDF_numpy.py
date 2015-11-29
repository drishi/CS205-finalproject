'''
Numpy serial implemnation of TFIDF
'''
import numpy as np

tf_vectors = None
question_texts = None
word_indices = None
tfidf_vectors = None
tfidf_norms = None
idf_vector = None
def load_questions(questions) :
  '''
  Load questions into memory
  '''
  global question_texts
  question_texts = questions

def load_indices(indices) :
  '''
  Load word indices into memory
  '''
  global word_indices
  word_indices = indices

def create_tfs() :
  '''
  Create Term Frequencies for questions
  '''
  global tf_vectors, question_texts
  tf_vectors = []
  for text in question_texts :
    vector = np.zeros(len(word_indices))
    for word in text :
      if word in word_indices :
        vector[word_indices[word]] += 1
    tf_vectors.append(vector)
  tf_vectors = np.array(tf_vectors)

def get_tfs() :
  '''
  Return Term Frequencies for questions
  '''
  global tf_vectors
  return tf_vectors

def calculate_tfidfs() :
  global question_texts, word_indices, tfidf_vectors, tfidf_norms, idf_vector
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
  tfidf_vectors = tf_vectors * idf_vector[None, :]
  tfidf_norms = np.linalg.norm(tfidf_vectors, axis=1)

def calculate_tfidf(example_question) :
  global word_indices
  tf_vector = np.zeros(len(word_indices))
  for word in example_question :
    if word in word_indices :
      tf_vector[word_indices[word]] += 1
  return tf_vector * idf_vector

def calculate_cossim(example_question) :
  global tfidf_vectors, tfidf_norms
  example = calculate_tfidf(example_question)
  return tfidf_vectors.dot(example) / (np.linalg.norm(example) * tfidf_norms)
