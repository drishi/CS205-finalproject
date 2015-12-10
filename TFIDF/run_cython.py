import cPickle as pickle
import numpy as np
import sys
import os.path
import time
sys.path.append(os.path.join('..', 'TFIDF'))
sys.path.append(os.path.join('', 'TFIDF'))
sys.path.append(os.path.join('..', 'util'))
sys.path.append(os.path.join('', 'util'))

import TFIDF_cython as tfidf_c

questions = pickle.load(open('data/questions_sm.pkl', 'rb'))
word_indices = pickle.load(open('data/wordIndices_sm.pkl', 'rb'))
question_texts = pickle.load(open('data/questionTexts_sm.pkl', 'rb'))

import sys

from sklearn.cluster import DBSCAN
import webbrowser

def displayCluster(db, how_near=2, how_many=3, label=0, max_pages=5):
    all_one_labels = []
    counter = 0
    for i in xrange(len(db.labels_)):
        if db.labels_[i] == -1:
            counter += 1
        elif db.labels_[i] == label:
            all_one_labels.append(i)
    counter = 0 
    for q in all_one_labels:
        lnk = questions[q]['link']
        if counter >= max_pages:
            break
        counter += 1
        webbrowser.open_new(lnk)
        time.sleep(3)

num_threads = raw_input('Number of Threads (number) [Default 4]: ')
if num_threads == '' :
  num_threads = 4
num_threads = int(num_threads)

while True :
  use_AVX = raw_input('use AVX (y/n) [Default y]: ')
  if use_AVX == 'y' :
    use_AVX = True
    break
  elif use_AVX == 'n' :
    use_AVX = False
    break
  elif use_AVX == '' :
    use_AVX = True
    break

while True :
  sim_hash = raw_input('Hash Size (32/64) [Default 64]: ')
  if sim_hash == '' :
    sim_hash = 64
  sim_hash = int(sim_hash)
  if sim_hash == 64 or sim_hash == 32 :
    break

num_locks = raw_input('Number of Locks (number) [Default 20000]: ')
if num_locks == '' :
  num_locks = 2000
num_locks = int(num_locks)


tfidf_c.init_globals(num_threads, use_AVX, sim_hash, num_locks)
tfidf_c.load_questions(question_texts)
tfidf_c.load_indices(word_indices)
tfidf_c.init_tfs()
tfidf_c.create_tfs()
tfidf_c.calculate_idf()
tfidf_c.init_tfidfs()
tfidf_c.calculate_tfidfs()
tfidf_c.calculate_simhashes()
cython_distances = np.asarray(tfidf_c.calculate_distances())
tfidf_c.cleanup()

# Create clusters
print "Calculating Clusters" 
db64 = DBSCAN(metric="precomputed", eps=1, min_samples=3).fit(cython_distances)

print "Number of documents: " + str(sum(map(lambda x: 1 if x >= 0 else 0, db64.labels_)))

labels = set(db64.labels_)

if -1 in labels: 
  labels.remove(-1)

print "Labels: "  + str(labels)


while True:
  label = raw_input('Which label to display? : ')
  
  # Super lazy way of handling input
  try :
    label = int(label)
    if label in labels :
      break
  except :
    continue

displayCluster(db64, label=label)

# all_one_labels = []
# for i in xrange(len(db64.labels_)):
#     if db64.labels_[i] == label:
#         all_one_labels.append(i)
# print all_one_labels

# for i in all_one_labels: 
#     print " ".join(question_texts[i])
#     print ""