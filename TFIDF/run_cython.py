import cPickle as pickle
import numpy as np
import sys
import os.path
sys.path.append(os.path.join('..', 'TFIDF'))
sys.path.append(os.path.join('', 'TFIDF'))
sys.path.append(os.path.join('..', 'util'))
sys.path.append(os.path.join('', 'util'))

import TFIDF_cython as tfidf_c

try :
    word_indices
except NameError:
    word_indices = pickle.load(open('wordIndices_sm.pkl', 'rb'))

try :
    question_texts
except NameError:
    question_texts = pickle.load(open('questionTexts_sm.pkl', 'rb'))

# Preprocess for cython code
tfidf_c.init_globals(1, True, "coarse", 64)
tfidf_c.load_questions(question_texts)
tfidf_c.load_indices(word_indices)
tfidf_c.init_tfs()
tfidf_c.create_tfs()
tfidf_c.calculate_idf(1)
tfidf_c.init_tfidfs()
tfidf_c.calculate_tfidfs()
tfidf_c.calculate_simhashes()
# tfidf.calculate_cossim(question_texts[0])

# Verification
# for (key, value) in word_indices :
#     if tfidf_c.get_value_at_key(key) != value :
#         print key
    