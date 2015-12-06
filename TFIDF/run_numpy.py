import cPickle as pickle
import numpy as np
import sys
import os.path
sys.path.append(os.path.join('..', 'TFIDF'))
sys.path.append(os.path.join('', 'TFIDF'))
sys.path.append(os.path.join('..', 'util'))
sys.path.append(os.path.join('', 'util'))

import TFIDF_numpy as tfidf

try:
     tfidf = reload(tfidf)
except NameError:
    import TFIDF_numpy as tfidf

try :
    word_indices
except NameError:
    word_indices = pickle.load(open('wordIndices_md.pkl', 'rb'))

try :
    question_texts
except NameError:
    question_texts = pickle.load(open('questionTexts_smd.pkl', 'rb'))
    
tfidf.init_globals()
tfidf.load_questions(question_texts)
tfidf.load_indices(word_indices)
tfidf.init_tfs()
serial_tfs = tfidf.create_tfs()
serial_idf = tfidf.calculate_idf()
serial_tfidfs = tfidf.calculate_tfidfs()
tfidf.calculate_tfidf_norms()
