import numpy as np

# This function assumes that W = [0,1,2,...,n_w-1]
def wt_from_Theta(Phi, w):
    wt_dist = Phi[:,w]
    wt_dist = wt_dist / np.sum(wt_dist)
    return wt_dist

def update_Theta(z, Theta, alpha):
    for i in range(Theta.shape[0]):    # document index
        for t in range(Theta.shape[1]):    # topic index
            n_it = np.sum(z[i]==t)
            Theta[i][t] = (n_it+alpha)/(z.shape[1]+Theta.shape[1]*alpha)
        Theta[i] = Theta[i] / np.sum(Theta[i])
    return Theta

def update_Phi(z, Phi, X, n_w, beta):
    for t in range(Phi.shape[0]):    # topic index
        for w in range(Phi.shape[1]):    # word index
            n_wt = np.sum((X==w)*(z==t))
            n = np.sum(X==w)
            Phi[t][w] = (n_wt+beta)/(n + n_w*beta)
        Phi[t] = Phi[t]/np.sum(Phi[t])
    return Phi