import numpy as np
import dist_gen

'''
Implementation adapted from Agustinus Kristiadi's Blog:
https://agustinus.kristia.de/techblog/2017/09/07/lda-gibbs/
----------------------
Symbols for all the parameters follow Griffiths et al 2004: 
https://www.pnas.org/content/pnas/101/suppl_1/5228.full.pdf?__=
T: Number of topics
W: The collection of all the words
n_w: Number of words
D: Number of documents

Theta ~ Dirichlet(alpha), document-topic distribution
Phi ~ Dirichlet(beta), topic-word distribution

X: corpus
Z: word-topic assignment

-- For Z --
n_ij_wi: the number of word wi assigned to topic j, not including the current one
n_ij_a:  the number of words assigned to topic j, not including the current one
n_ij_di: the number of words in document di assigned to topic j, not including the current one
n_i_di:  the number of words in di minus one

-- For Phi --
n_jw: The number of word w assigned to topic j
n_ja: The total number of word in topic j in z

-- For Theta --
n_jd: The number of words in document d assigend to j
n_ad: The number of words in document d
'''

def lda_gibbs_param_smart(D, T, W, Theta, Phi, X, Z, alpha, beta, iterations=1000):
    n_w = len(W)
    for it in range(iterations):
        for i in range(D):
            for v in range(n_w):
                p_iv = np.exp(np.log(Theta[i]) + np.log(Phi[:, X[i, v]]))
                p_iv /= np.sum(p_iv)
                Z[i, v] = np.random.multinomial(1, p_iv).argmax()

        # Sample from full conditional of Theta - document-topic distribution
        for d in range(D):
            for j in range(T):
                n_jd = np.sum(Z[d]==j)
                n_ad = X[d].shape[0]
                Theta[d][j] = (n_jd + alpha) / (n_ad + T*alpha)

        # Sample from full conditional of Phi - topic-word distribution
        for j in range(T):
            for w in range(n_w):
                n_jw = find_n_jw(Z, X, j, w)
                n_ja = np.sum(Z==j)
                Phi[j][w] = (n_jw + beta) / (n_ja + T*beta)

    return Theta, Phi, Z


def lda_gibbs_param(D, T, W, Theta, Phi, X, z, alpha, beta, iterations=1000):
    n_w = len(W)
    for it in range(iterations):
        for d in range(D):
            for w in range(n_w):
                z[d][w] = -1
                Theta = dist_gen.update_Theta(z, Theta, alpha)
                Phi = dist_gen.update_Phi(z, Phi, X, n_w, beta)
                this_word = X[d][w]
                topic_dist = Theta[d][:] * Phi[:, this_word]
                topic_dist = topic_dist / np.sum(topic_dist)
                z[d][w] = np.random.multinomial(1, topic_dist).argmax()
                # for j in range(T):
                #     n_ij_wi = find_n_ij_wi(z, X, j, w, d)   # nzw
                #     n_ij_a  = np.sum(z==j)-1 if z[d][w]==j else np.sum(z==j)    # nz
                #     n_ij_di = np.sum(z[d]==j)-1 if z[d][w]==j else np.sum(z[d]==j)   # nmz
                #     n_i_di  = X[d].shape[0]-1    # nm
                #     P_zdw[j] = (n_ij_wi + beta)/(n_ij_a + n_w*beta) * (n_ij_di+alpha)/(n_i_di+T*alpha)
                # P_zdw = P_zdw / np.sum(P_zdw)
                # z[d][w] = np.random.multinomial(1, P_zdw).argmax()
    Theta = dist_gen.update_Theta(z, Theta, alpha)
    Phi = dist_gen.update_Phi(z, Phi, X, n_w, beta)
    return Theta, Phi, z


def find_n_jw(Z, X, j, w):
    n_jw = 0
    for d in range(X.shape[0]):
        for i in range(X.shape[1]):
            if Z[d][i]==j and X[d][i]==w:
                n_jw+=1
    return n_jw

def find_n_ij_wi(Z, X, j, w, d):
    n_ij_wi = 0
    for di in range(X.shape[0]):
        for i in range(X.shape[1]):
            if di==d and i==w:
                continue
            elif Z[di][i]==j and X[di][i]==w:
                n_ij_wi+=1
    return n_ij_wi