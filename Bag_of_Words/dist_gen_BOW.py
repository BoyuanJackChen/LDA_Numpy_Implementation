import numpy as np

# This function assumes that W = [0,1,2,...,n_w-1]. wt short for word-topic
def wt_from_Phi(Phi, w):
    wt_dist = Phi[:,w]
    wt_dist = wt_dist / np.sum(wt_dist)
    return wt_dist

def update_Theta(Z, Theta, alpha):
    for d in range(Theta.shape[0]):    # document index
        for t in range(Theta.shape[1]):    # topic index
            n_td = sum([i[t] for i in Z[d]])
            n_d = sum(sum(Z[d], []))
            Theta[d][t] = (n_td + alpha) / (n_d + Theta.shape[1] * alpha)
        Theta[d] = Theta[d] / np.sum(Theta[d])  # Normalize on the document level
    return Theta


def update_Phi(Z, Phi, X, beta):
    for t in range(Phi.shape[0]):    # topic index
        for w in range(Phi.shape[1]):    # word index
            n_tw = get_n_tw(Z, X, t, w)
            n_ta = get_n_t(Z, t)
            Phi[t][w] = (n_tw + beta) / (n_ta + Phi.shape[0] * beta)
        Phi[t] = Phi[t]/np.sum(Phi[t])
    return Phi

def get_n_tw(Z, X, t, w):
    n_tw = 0
    for d in range(len(X)):
        for xi in range(len(X[d])):
            if X[d][xi][0]==w:
                n_tw += Z[d][xi][t]
                break
    return n_tw

def get_n_t(Z, t):
    n_t = 0
    for d in Z:
        for w in d:
            n_t += w[t]
    return n_t