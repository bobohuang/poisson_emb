'''
First created 2016 Aug
Author: Disi Ji
'''
import copy

def pairwordcontext(data):
    '''
    INPUT:
        data: csr matrix of data like "tw_oc"
    OUTPUT:
        same as INPUT of PoissonEmb
    this step is very inefficent for bag of words typepf context
    '''
    
    N  = len(data.data)
    I = data.shape[0]
    L = data.shape[1] 
    
    user_history = [None]*I
    words = [None]*N
    contexts = [None]*N
    n_w = [None]*N
      
    
    for i in range(I):
        value = data[i,:].data
        column_index = data[i,:].indices
        temp = dict()
        for j in range(len(value)):
            temp[column_index[j]] = value[j]
        user_history[i] = temp
        
    # make [word,context] pairs
    pos = 0
    for i in range(I):
        temp = user_history[i]
        for j in list(temp.keys()):
            words[pos] = j
            n_w[pos] = temp[j]
            contexts[pos] = copy.deepcopy(temp)
            del contexts[pos][words[pos]]
            pos += 1
    return words[0:pos-1],contexts[0:pos-1],n_w[0:pos-1],L