import numpy as np

'''
Computing LDA parameters using naive gibbs sampling
Return:
    Theta: document-topic distribution based on count
    Phi: word-topic distribution based on count
    z: word-topic assignment of the model, BOW format
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