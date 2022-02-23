import numpy as np
import LDA_BOW

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

# Changing document to BOW format
# Same as BOW format used by Gensim
X_new = []
for i in range(0,len(X)):
    doc = np.zeros(W.shape[0])
    for j in X[i]:
        doc[j] += 1
    doc2 = []
    for j in range(0,W.shape[0]):
        if doc[j] != 0:
            doc2.append((j,int(doc[j])))
    X_new.append(doc2)

X = X_new


# Necesary parameters
D = len(X)  # num of docs
n_w = W.shape[0]  # num of words
T = 2  # num of topics


# Dirichlet priors
alpha = 1/T   # Theta, document-topic
beta = 1/T    # Phi, topic-word
iterations = 1000



# z := word-topic assignment, aka real assignment to real words in document X. 
# z[i,j]: tuple of length T, containg the number of words assigned to each topic
# e.g. z[i,j][0]: num of words assigned to topic 0
z = []
for doc in X:     # document index
    z_doc = []
    for word in doc:     # word index
        w = word[0]
        count = word[1]
        assigned_topic = np.random.multinomial(count, [0.5]*T)
        z_doc.append(tuple(assigned_topic))
    z.append(z_doc)


# Theta := document-topic distribution
Theta = np.zeros((D,T))

# Phi := word-topic distribution
Phi = np.zeros((T,len(W)))

# Initializing Theta and Phi
z = []
for doc in X:     # document index
    z_doc = []
    for word in doc:     # word index
        w = word[0]
        count = word[1]
        assigned_topic = np.random.multinomial(count, [0.5]*T)
        z_doc.append(tuple(assigned_topic))
    z.append(z_doc)

for i in range(0,D):
    for j in range(0,len(X[i])):
        w = X[i][j][0]
        count = X[i][j][1]     
        c_z = z[i][j]
        for k in range(0,T):
            Theta[i][k] += c_z[k]
            Phi[k, w] += c_z[k]

# Run LDA
Theta, Phi, z = LDA_BOW.lda_gibbs_BOW(D, T, W, Theta, Phi, X, z, alpha, beta, iterations=2000)

print(Theta)
print(Phi)
print(z)
print(X)


