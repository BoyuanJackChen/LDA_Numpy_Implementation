import numpy as np
import dist_gen_BOW as dist_gen

'''
Computing LDA parameters using naive gibbs sampling
Return:
    Theta: document-topic distribution based on count
    Phi: word-topic distribution based on count
    z: word-topic assignment of the model, BOW format

-- For Z --
n_ij_wi: the number of word wi assigned to topic j, not including the current one
n_ij_a:  the number of words assigned to topic j, not including the current one
n_ij_di: the number of words in document di assigned to topic j, not including the current one
n_i_di:  the number of words in di minus one

-- For Phi --
n_jw: The number of word w assigned to topic j
n_ja: The total number of word in topic j in z

-- For Theta --
n_td: The number of words in document d assigend to topic t
n_d: The number of words in document d
'''

def lda_gibbs_BOW(D, T, W, Theta, Phi, X, z, alpha, beta, iterations=2000):
    n_w = len(W)
    # An additional word topic varaible that helps us
    # compute the multinomial probability
    word_topic = np.zeros(T)
    for i in range(0,D):
        for j in range(0,len(X[i])):
            c_z = z[i][j]
            for k in range(0,T):
                word_topic[k] += c_z[k]

    # Naive gibbs sampling to calculate theta and phi
    for h in range(0,iterations):
        for i in range(0,D):

            # Finding the length of the document
            doc_len = sum([j[1] for j in X[i]])
            for j in range(0,len(X[i])):    

                # Word/count/topic to be examined
                w = X[i][j][0]
                count = X[i][j][1]     
                c_z = z[i][j]

                # Remove the current word count from the parameters
                for k in range(0,T):
                    Theta[i][k] -= c_z[k]
                    Phi[k, w] -= c_z[k]
                    word_topic[k] -= c_z[k]

                # Computing the probabilities
                prob = ((Theta[i]+alpha) / (doc_len-1+T*alpha)) * \
                    ((Phi[:,w] + beta) / (word_topic +n_w*beta))
                n_z = np.random.multinomial(count, prob/sum(prob))

                # Reassigning the word/topic and updating parameters
                z[i][j] = tuple(n_z)
                c_z = z[i][j]
                for k in range(0,T):
                    Theta[i][k] += c_z[k]
                    Phi[k, w] += c_z[k]
                    word_topic[k] += c_z[k]
    # Normalize Theta and Phi
    for i in range(len(Theta)):
        Theta[i] = Theta[i] / sum(Theta[i])
    for i in range(len(Phi)):
        Phi[i] = Phi[i] / sum(Phi[i])
    return Theta, Phi, z


# In this function, Theta and Phi are always probability distributions.
def lda_gibbs_collapsed_BOW(n_d, n_t, W, Theta, Phi, X, Z, alpha, beta, iterations=2000):
    n_w = len(W)
    for it in range(iterations):
        for d in range(n_d):
            for xi in range(len(X[d])):  # Iterate over the BOW tuples in this document
                Z[d][xi] = [0] * n_t
                Theta = dist_gen.update_Theta(Z, Theta, alpha)
                Phi = dist_gen.update_Phi(Z, Phi, X, beta)
                # Get the distribution of this word in this document
                w, count = X[d][xi]  # The word index, and the number of that word in this document
                p_dw = np.exp(np.log(Theta[d]) + np.log(Phi[:, w]))
                p_dw /= np.sum(p_dw)
                assigned_topics = np.random.multinomial(count, p_dw)
                Z[d][xi] = assigned_topics.tolist()
    Theta = dist_gen.update_Theta(Z, Theta, alpha)
    Phi = dist_gen.update_Phi(Z, Phi, X, beta)
    return Theta, Phi, Z
