# this is an implementation for exponential family embeddings
# here we start from Poisson embedding
'''
Author: Disi Ji
'''
import numpy as np
from math import *
import matplotlib.pyplot as plt


def PoissonEmb(words,contexts,n_w,N,K=2):
    
    '''
    INPUT:
        words: a flat list of [[word in article] article in corpus]
        context: [(c: n_c) for c[word] in words], a list of dictionaries
        n_w: [count of w for w in words], of same shape as words
        N: number of words that embeddings and context embeddings should be learned
        K: dimention of embeddings
    OUTPUT:
        rho: N * K matrix of item embeddings
        alpha: N * K matrix of context embeddings, normalized by row
    '''
    
    def contextvector(context,alpha):
        # context is a dict
        if len(context)==0:
            return np.matrix(np.zeros(alpha.shape[1]))
        temp = np.array([alpha[i] for i in context.keys()])
        cv = np.matrix(list(context.values())) * temp
        return cv
    
    def PlotEmb2d(rho,alpha,data=1):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.2), sharey=True)
        colors = [i for i in range(rho.shape[0])]
        ax1.scatter(rho[:,0], rho[:,1], s=data, c=colors, alpha=0.5)
        ax1.set_title('rho')
        ax2.scatter(alpha[:,0], alpha[:,1], s=data, c=colors, alpha=0.5)
        ax2.set_title('alpha')
        plt.show()

    ######hold out 1% data to check convergence#####
    L = len(words)
    l = int(L/100)
    val = np.random.choice(L, l)
    words_val = [words[i] for i in val]
    contexts_val = [contexts[i] for i in val]
    n_w_val = [n_w[i] for i in val]
    for i in sorted(val, reverse=True):
        del words[i]
        del contexts[i]
        del n_w[i]
        
        
    ######set parameters#####
    L = len(words)
    l = int(L/10) # size of subsample
    mu, sigma = 0, 0.1
    maxiter = 11
    lambd = 100
    step = 0.01
    likelihood = -inf # log likelihood
    epsilon = 0.0001
    
    
    ######store idx of the words and contexts######
    idx = dict() # idx[i]: idx of (word, context) pairs where word == i
    invidx = dict() # invidx[i] = idx of (word, context) pairs where i in context

    for i in range(N):
        idx[i] = []
        invidx[i] = []

    for i in range(len(words)):
        idx[words[i]].append(i)
        for j in contexts[i]:
            invidx[j].append(i)
        
               
    ######initialize the rho and alpha matrix######
    rho = np.asmatrix(np.random.normal(mu, sigma, (N,K)))
    alpha = np.asmatrix(np.random.normal(mu, sigma, (N,K)))
    
    PlotEmb2d(rho,alpha)

    ######update rho and alpha with gradient descent#######
    likelihood_traj = []
    likelihood_traj.append(likelihood)

    for iter in range(maxiter):
        
        # check for convergence
        likelihood = -lambd/2*((np.square(rho)).sum()+(np.square(alpha)).sum())
        for i in range(len(val)):
            v = contextvector(contexts_val[i],alpha)
            ita = (rho[words[i]]*(v.T))[0,0]
            rate = exp(ita)
            x = n_w_val[i]
            likelihood += x*ita - rate
        print('Log likelihood of iteration %d: %.6f' % (iter, likelihood/len(words_val)))
        if abs(likelihood_traj[-1] - likelihood) < epsilon:
            break
        likelihood_traj.append(likelihood)
        

        samples = np.random.choice(L, l)
        
        # update rho
        for n in range(N):
            idx_samples = [val for val in idx[n] if val in samples]
            if len(idx_samples)==0:
                continue
            gradient = -lambd*rho[n]*0.1
            for i in idx_samples:
                v = contextvector(contexts[i],alpha)
                ita = (rho[words[i]]*(v.T))[0,0]
                gradient += (n_w[i] - exp(ita))*v
            rho[n] += step*gradient/np.linalg.norm(gradient)
        
            
        # update alpha
        for n in range(N):
            idx_samples = [val for val in invidx[n] if val in samples]
            if len(idx_samples)==0:
                continue
            gradient = -lambd*alpha[n]*0.1
            for i in idx_samples:
                v = contextvector(contexts[i],alpha)
                ita = (rho[words[i]]*(v.T))[0,0]
                gradient += (n_w[i] - exp(ita))*rho[words[i]]*contexts[i][n]
            alpha[n] += step*gradient/np.linalg.norm(gradient)

        PlotEmb2d(rho,alpha)

    print(len(likelihood_traj))

    #####plot log likelihood######
    plt.plot(likelihood[1:,])
    plt.title('Log likelihood on hold out data')
    plt.show()  

    return rho,alpha