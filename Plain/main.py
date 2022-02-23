import numpy as np
import lda
import dist_gen

'''
Implementation adapted from Agustinus Kristiadi's Blog:
https://agustinus.kristia.de/techblog/2017/09/07/lda-gibbs/
----------------------
All the parameters correspond to Finding scientific topics 2004. 
T: Number of topics
W: The collection of all the words
n_w: Number of words
D: Number of documents

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

# Document words
X = np.array([
    [0, 0, 1, 2, 2],
    [0, 0, 1, 1, 1],
    [0, 1, 2, 2, 2],
    [2, 2, 1, 1, 4],
    [4, 4, 4, 4, 4],
    [3, 3, 4, 4, 4],
    [3, 4, 4, 4, 4],
    [3, 3, 3, 4, 1],
    [4, 4, 3, 3, 2],
])

D = X.shape[0]  # num of docs
n_w = W.shape[0]  # num of words
T = 2  # num of topics


'''Randomized Initialization'''
# Dirichlet priors
alpha = 1   # Theta, document-topic
beta = 1    # Phi, topic-word
iterations = 1000

# Theta := document-topic distribution
Theta = np.zeros([D, T])
for i in range(D):
    Theta[i] = np.random.dirichlet(alpha*np.ones(T))

# Phi := word-topic distribution
Phi = np.zeros([T, n_w])
for k in range(T):
    Phi[k] = np.random.dirichlet(beta*np.ones(n_w))

# z := word-topic assignment, aka real assignment to real words in document X. 
#      Initialize here in original Theta distributions.
z = np.zeros(shape=X.shape, dtype=int)
for i in range(z.shape[0]):     # document index
    for j in range(z.shape[1]):     # word index
        w = X[i][j]
        wt_dist = dist_gen.wt_from_Theta(Phi, w)
        z[i][j] = np.random.multinomial(1, wt_dist).argmax()

# # If you use inference method, you can assign them randomly at first. You don't need to use
# # this variable if implemented with gibbs sampling. 
# Z = np.zeros(shape=[D, n_w], dtype=int)
# for i in range(D):
#     for l in range(n_w):
#         Z[i, l] = np.random.randint(T)  # randomly assign word's topic

# Run LDA. 
Theta, Phi, z = lda.lda_gibbs(D, T, W, Theta, Phi, X, z, alpha, beta, iterations=2000)
print(Theta)
print(Phi)
print(z)
print(X)

# Your goal is to give me a Theta file like this after 2000 iterations:
# [[0.14285714 0.85714286]
#  [0.14285714 0.85714286]
#  [0.28571429 0.71428571]
#  [0.42857143 0.57142857]
#  [0.85714286 0.14285714]
#  [0.71428571 0.28571429]
#  [0.85714286 0.14285714]
#  [0.85714286 0.14285714]
#  [0.71428571 0.28571429]]