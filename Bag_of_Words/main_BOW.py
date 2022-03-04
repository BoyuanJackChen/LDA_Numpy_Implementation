import numpy as np
import LDA_BOW
import dist_gen_BOW as dist_gen

'''
Implementation adapted from Agustinus Kristiadi's Blog:
https://agustinus.kristia.de/techblog/2017/09/07/lda-gibbs/
----------------------
All the parameters correspond to Finding scientific topics 2004. 
n_t: Number of topics
W: The collection of all the words
n_w: Number of words in dictionary
n_d: Number of documents

Theta ~ Dirichlet(alpha), document-topic distribution
Phi ~ Dirichlet(beta), topic-word distribution

X: corpus
Z: word-topic assignment, shape (D,n_w)

n_jw: The number of word w assigned to topic j in Z
n_ja: The total number of word in topic j in Z
n_jd: The number of words in document d assigend to j
n_ad: The number of words in document d 
'''

# Vocabulary - all the words
W = np.array([0, 1, 2, 3, 4])

# Toy sample documents
X = [
    [0, 0, 1, 2, 2],
    [0, 0, 1, 1, 1],
    [0, 1, 2, 2, 2],
    [2, 2, 1, 1, 4],
    [4, 4, 4, 4, 4],
    [3, 3, 4, 4, 4],
    [3, 4, 4, 4, 4],
    [3, 3, 3, 4, 1],
    [4, 4, 3, 3, 2],
]
# X in BOW format:
# [[(0, 2), (1, 1), (2, 2)],
#  [(0, 2), (1, 3)],
#  [(0, 1), (1, 1), (2, 3)],
#  [(1, 2), (2, 2), (4, 1)],
#  [(4, 5)],
#  [(3, 2), (4, 3)],
#  [(3, 1), (4, 4)],
#  [(1, 1), (3, 3), (4, 1)],
#  [(2, 1), (3, 2), (4, 2)]]


# Changing document to BOW format
# Same as BOW format used by Gensim
X_BOW = []
for i in range(len(X)):
    doc = np.zeros(W.shape[0])
    for j in X[i]:
        doc[j] += 1
    doc2 = []
    for j in range(W.shape[0]):
        if doc[j] != 0:
            doc2.append((j,int(doc[j])))
    X_BOW.append(doc2)
X = X_BOW


# Necesary parameters
n_d = len(X)  # num of docs
n_w = W.shape[0]  # num of words
n_t = 2  # num of topics


# Dirichlet priors
alpha = 1/n_t   # Theta, document-topic
beta = 1/n_t    # Phi, topic-word
iterations = 1000


# Theta := document-topic distribution. Initialized in Dirichlet(alpha)
Theta = np.zeros((n_d,n_t))
for i in range(n_d):
    Theta[i] = np.random.dirichlet(alpha*np.ones(n_t))

# Phi := word-topic distribution. Initialized in Dirichlet(beta)
Phi = np.zeros((n_t,n_w))
for k in range(n_t):
    Phi[k] = np.random.dirichlet(beta*np.ones(n_w))

# Z := word-topic assignment, aka real assignment to real words in corpus X.
# Z[i,j]: tuple of length T, containg the number of words assigned to each topic
# e.g. Z[i,j][0]: num of words assigned to topic 0 in document i word j from corpus X
Z = []
for doc in X:     # document index
    Z_doc = []
    for word in doc:     # word index
        w = word[0]
        count = word[1]
        wt_dist = dist_gen.wt_from_Phi(Phi, w)
        assigned_topic = np.random.multinomial(count, wt_dist)
        Z_doc.append(assigned_topic.tolist())
    Z.append(Z_doc)
Theta, Phi, Z = LDA_BOW.lda_gibbs_collapsed_BOW(n_d, n_t, W, Theta, Phi, X, Z, alpha, beta, iterations=3000)

# Theta = np.zeros((n_d,n_t))
# Phi = np.zeros((n_t,len(W)))
# for i in range(n_d):
#     for j in range(0,len(X[i])):
#         w = X[i][j][0]
#         count = X[i][j][1]
#         c_z = Z[i][j]
#         for k in range(0,n_t):
#             Theta[i][k] += c_z[k]
#             Phi[k, w] += c_z[k]
# Theta, Phi, Z = LDA_BOW.lda_gibbs_BOW(n_d, n_t, W, Theta, Phi, X, Z, alpha, beta, iterations=2000)

print("Theta is: ")
print(Theta)
print()
print("Phi is:")
print(Phi)
print()
print("z and X are:")
print(Z)
print(X)
