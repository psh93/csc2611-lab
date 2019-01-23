from nltk.corpus import brown
from nltk import ngrams, FreqDist
from scipy import sparse, stats
from gensim import matutils
from gensim.models import LsiModel
import numpy as np
import math

########################### Part 1 ############################

## Step 2 ##

# Construct a unigram model on the Brown corpus.
# n = 5000 most "common" words from the corpus are stored in W.          
btokens = brown.words()
fdist = FreqDist(btokens)
n = 5000
W = [ word[0] for word in fdist.most_common(n) ]

# Print 5 most and least common words.
print("5 most common words:\n")
for word in W[:5]:
    print(word)

print("5 least common words:\n")
for word in W[-5:]:
    print(word)

# Update W by adding words in Table1 of RG65.
f = open('RG65.txt', 'r')
P = []
S = []

for line in f.readlines():

    line = line.split('\t')
    w1 = line[0]
    w2 = line[1]

    # Find all pairs of words found in W for Step 6.
    if w1 in W and w2 and W:
        P.append((w1, w2))
        S.append(float(line[2]))

    # Ignore the scores(third column), and add the words.
    W.extend(w for w in line[:2] if w not in W)

## Step 3 ## 

# Retrieve bigram counts in the Brown corpus.
bg_brown = list(ngrams(brown.words(), 2))
fdist = FreqDist(bg_brown)

# For each bigram, construct a word-context model by
# recording the counts at row = word and col = context.
counts = []
indices = []
for i, word in enumerate(W):
    for j, context in enumerate(W):

        c = fdist.get((word, context))
        
        if c == None: continue
        counts.append(c)
        indices.append([i,j])

rows = np.array(indices)[:, 0]
cols = np.array(indices)[:, 1]
l = len(W)

# For efficiency, store the data in a sparse matrix form.
M1 = sparse.coo_matrix((counts, (rows, cols)), shape = (l,l))

## Step 4 ## 

# Compute ppmi on M1 from Step 2.
total = M1.sum()
ppmi = []
history = {}
for i,j,count in zip(M1.row, M1.col, M1.data):

    if count in history:

        ppmi.append(count)
        continue
    
    jointProb = M1.getrow(i).getcol(j).data[0] / total 
    Pi = M1.getrow(i).sum() / total
    Pj = M1.getcol(j).sum() / total

    pmi = math.log(jointProb / (Pi * Pj), 2)
    val = max(pmi, 0)

    ppmi.append(val)
    history[count] = ppmi

# Again, the data is stored in a sparse matrix.
M1Plus = sparse.coo_matrix((ppmi, (rows, cols)), shape = (l,l))

## Step 5 ##

# To use gensim's implementation of LSA model, the matrix first
# has to be transformed using their Sparse2Corpus method.
gensim_matrix = matutils.Sparse2Corpus(M1Plus)
lsa_models = {}

# Dimensionalities for reduction.
dim = [10, 100, 300]
for d in dim:

    model = LsiModel(gensim_matrix, num_topics = d)
    lsa_models[d] = model.projection.u
    print("LSA decomp. --> %s" % (lsa_models[d].shape,))

## Step 6 ##

# Definition of a cosine function.
def cosine(v1, v2):

    dp = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return 0

    return dp / (n1 * n2)

# For each model, compute cosine similarities between
# the pairs of words found from Step6. 
S1, S2 = [], []
S3 = dict()
for wi, wj in P:

    # Retrieve locations of the words in W.
    i = W.index(wi)
    j = W.index(wj)

    # M1: word-context
    v1 = M1.getrow(i).toarray()[0]
    v2 = M1.getrow(j).toarray()[0]

    S1.append(cosine(v1, v2))

    # M1+: ppmi    
    v1 = M1Plus.getrow(i).toarray()[0]
    v2 = M1Plus.getrow(j).toarray()[0]

    S2.append(cosine(v1, v2))

    # M2: LSA in 10, 100, 300 dim.
    for d, model in zip(dim,lsa_models.values()):

        if d not in S3.keys():
            S3[d] = []

        v1 = model[i]
        v2 = model[j]

        score = cosine(v1, v2)
        S3[d].append(score)

## Step 8 ##

# Compute Pearson correlations between the scores from Step 7.
print("Pearson corr\n")
print("word-context model: %.4f" % stats.pearsonr(S, S1)[0])
print("ppmi model: %.4f" % stats.pearsonr(S, S2)[0])
for d in dim:

    pscores = S3[d]
    print("LSA (dim = %d): %.4f" % (d, stats.pearsonr(S, pscores)[0]))

########################### Part 2 ############################

from gensim.models import KeyedVectors
bpath = './GoogleNews-vectors-negative300.bin'
apath = './word-test.txt'

## Step 2, 3 ##

# Load and store the pre-trained vectors.
model = KeyedVectors.load_word2vec_format(bpath, binary = True)

# Compute the cosine similarities between the same pairs of words
# used for analysis from Part 1, using the above Word2Vec model.
S4 = []
for wi, wj in P:

    v1 = model[wi]
    v2 = model[wj]

    score = cosine(v1, v2)
    S4.append(score)

# Compute a Pearson correlation.
print("word2vec model: %.4f" % stats.pearsonr(S, S4)[0])

## Step 4 ## 

# Perform an analogy test.
analogy_scores = model.wv.evaluate_word_analogies(apath)

# Perform the same analogy test with LSA, but with a subset of
# tests only, because some words don't exist in W.
valid_tests = []
for line in open(apath, 'r').readlines():

    tokens = line.split()
    if tokens[0] == ":": continue
    
    c = 1
    for word in tokens:

        if word not in W: break
        c = c + 1

    if c > 4:

        valid_tests.append(tokens)
        c = 1

# Find the vector(word) that is closest to the given vector.
def most_similar(v1, d):

    min_cos = 10
    min_i = 0 
    for i in range(len(W)):

        v2 = lsa_models[d][i]
        similarity = cosine(v1, v2)

        if min_cos > similarity:

            min_cos = similarity
            min_i = i

    return min_i

# Average score is returned after counting correct predictions.
def accuracy(pred, target):

    total = len(pred)
    score = 0

    if len(pred) == len(target):
        score = sum(1 for w1, w2 in zip(pred, target) if w1 == w2)

    return score / total

# Analogy test for LSA model in 300 dimensions.
solset = [ test[3] for test in valid_tests ]
pred = []
model = lsa_models[300]
    
for test in valid_tests:
    
    solset.append(test[3])

    i = W.index(test[0])
    j = W.index(test[1])
    k = W.index(test[2])

    v1 = model[i]
    v2 = model[j]
    v3 = model[k]

    # For this analogy test, the resultant vector is compared with
    # all vector representations of words in W to find the closest one.
    vpred = (v1 + v2) - v3
    idx = most_similar(vpred, 300)
    wpred = W[idx]

    # Store the predicted words.
    pred.append(wpred)

acc = accuracy(pred, solset)
print(acc)
