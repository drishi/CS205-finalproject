import cPickle as pickle
import numpy as np
import sys
import os.path
sys.path.append(os.path.join('..', 'TFIDF'))
sys.path.append(os.path.join('', 'TFIDF'))
sys.path.append(os.path.join('..', 'util'))
sys.path.append(os.path.join('', 'util'))

import tfidf_numpy as tfidf

word_indices = pickle.load(open('data/wordIndices_sm.pkl', 'rb'))

question_texts = pickle.load(open('data/questionTexts_sm.pkl', 'rb'))
    
tfidf.init_globals()
tfidf.load_questions(question_texts)
tfidf.load_indices(word_indices)
tfidf.init_tfs()
serial_tfs = tfidf.create_tfs()
serial_idf = tfidf.calculate_idf()
serial_tfidfs = tfidf.calculate_tfidfs()
serial_cossims = tfidf.calculate_cossims()
serial_simhashes = tfidf.calculate_simhashes()
serial_distances = tfidf.calculate_distances()